import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, dump_tensor_to_mat
from policies.pruners import GradualPruner
import math

class WoodburryFisherPruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(WoodburryFisherPruner, self).__init__(model, inp_args, **kwargs)
        print("IN WOODBURRY")
        self._fisher_inv_diag = None
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        if self.args.woodburry_joint_sparsify:
            self._param_stats = []
        if self.args.dump_fisher_inv_mat:
            self._all_grads = []

    def _compute_sample_fisher(self, loss, return_outer_product=True):

        ys = loss
        params = []
        for module in self._modules:
            for name, param in module.named_parameters():
                # print("name is {} and shape of param is {} \n".format(name, param.shape))

                if self._weight_only and 'bias' in name:
                    continue
                else:
                    params.append(param)

        grads = torch.autograd.grad(ys, params)  # first order gradient

        # Do gradient_masking: mask out the parameters which have been pruned previously
        # to avoid rogue calculations for the hessian

        for idx, module in enumerate(self._modules):
            grads[idx].data.mul_(module.weight_mask)

        grads = flatten_tensor_list(grads)

        if self.args.dump_grads_mat:
            self._all_grads.append(grads)

        self._num_params = len(grads)

        if not return_outer_product:
            return grads
        else:
            return torch.ger(grads, grads)

    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params):
        return -1 * torch.div(torch.mul(pruned_params, flattened_params), self._fisher_inv_diag)

    @staticmethod
    def _get_param_stat(param, param_mask, fisher_inv_diag, param_idx):
        if param is None or param_mask is None: return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = fisher_inv_diag[param_idx: param_idx + param.numel()].view_as(param)
        inv_fisher_diag_entry = inv_fisher_diag_entry.to(param.device)
        print("mean value of statistic without eps = {} is ".format(1e-10),
              torch.mean((param ** 2) / inv_fisher_diag_entry))
        print(
            "std value of statistic without eps = {} is ".format(1e-10),
            torch.std((param ** 2) / inv_fisher_diag_entry))
        return ((param ** 2) / (inv_fisher_diag_entry + 1e-10) + 1e-10) * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2):
        piece = int(math.ceil(len(vec) / num_parts))
        vec_len = len(vec)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                    torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)],
                              vec[j * piece:min((j + 1) * piece, vec_len)])
                )

    def _compute_woodburry_fisher_inverse(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        self._model = self._model.to(device)

        print("in woodfisher: len of subset_inds is ", len(subset_inds))

        goal = self.args.fisher_subsample_size

        assert len(subset_inds) == goal * self.args.fisher_mini_bsz

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=self._fisher_mini_bsz, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        if self.args.disable_log_soft:
            # set to true for resnet20 case
            # set to false for mlpnet as it then returns the log softmax and we go to NLL
            criterion = torch.nn.functional.cross_entropy
        else:
            criterion = F.nll_loss

        self._fisher_inv = None

        num_batches = 0
        num_samples = 0

        for in_tensor, target in dummy_loader:
            self._release_grads()

            in_tensor, target = in_tensor.to(device), target.to(device)
            output = self._model(in_tensor)
            loss = criterion(output, target)
            # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
            # which is just a practical heuristic to utilize more datapoints

            sample_grads = self._compute_sample_fisher(loss, return_outer_product=False)
            if self.args.fisher_cpu:
                sample_grads = sample_grads.cpu()

            if aux_device is not None and aux_device != torch.device('cpu'):
                sample_grads = sample_grads.to(aux_device)

            # print(f'device of sample_grads is {sample_grads.device}')
            if num_batches == 0:

                numerator_normalization = (self.args.fisher_damp) ** 2

                # rewrite in terms of inplace operations
                self._fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                    goal + (sample_grads.dot(sample_grads) / self.args.fisher_damp)
                )
                self._fisher_inv.diagonal().sub_(1.0 / self.args.fisher_damp)
                # 1/self.args.fisher_damp \times Identity matrix is used to represent (H^-1)_0
                self._fisher_inv.mul_(-1)

            else:
                cache_matmul = torch.matmul(self._fisher_inv, sample_grads)
                cache_matmul.div_((goal + sample_grads.dot(cache_matmul)) ** 0.5)
                if not self.args.fisher_optimized:
                    self._fisher_inv.sub_(
                        torch.ger(cache_matmul, cache_matmul)
                    )
                else:
                    assert self.args.fisher_parts > 1
                    # F = F - x x'
                    # F1 = -F
                    self._fisher_inv.mul_(-1)
                    # F1 + x x'
                    self._add_outer_products_efficient_v1(
                        self._fisher_inv, cache_matmul, num_parts=self.args.fisher_parts
                    )
                    # F = - F1
                    self._fisher_inv.mul_(-1)

                del cache_matmul

                del sample_grads

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            # print("# of examples done {} and the goal is {}".format(num, goal))

            if num_samples == goal * self._fisher_mini_bsz:
                break

        print("# of examples done {} and the goal (#outer products) is {}".format(num_samples, goal))
        print("# of batches done {}".format(num_batches))
        self._fisher_inv_diag = self._fisher_inv.diagonal()

        end_time = time.perf_counter()
        print("Time taken to compute fisher inverse with woodburry is {} seconds".format(str(end_time - st_time)))

        if self.args.dump_fisher_inv_mat:
            dump_tensor_to_mat(self._fisher_inv.diagonal(), self.args.run_dir, 'fisher_inv_diag.mat', 'fisher_inv_diag')

        if self._inspect_inv:
            print('---- Inspecting fisher inverse ----')
            inspect_dic = get_summary_stats(self._fisher_inv)
            inspect_dic['trace'] = self._fisher_inv.trace().item()
            inspect_dic['sum'] = self._fisher_inv.sum().item()
            inspect_dic['trace/sum'] = (self._fisher_inv.trace() / self._fisher_inv.sum()).item()
            self.inspect_dic = inspect_dic
            print(self.inspect_dic)
            print('-----------------------------------')

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            print("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if they were deleted during gradual pruning
        if not hasattr(self, '_all_grads'):
            self._all_grads = None
        if not hasattr(self, '_param_stats'):
            self._param_stats = []

        #############################################################
        # Step 1. Computer full fisher inverse via woodburry
        self._compute_woodburry_fisher_inverse(dset, subset_inds, device, num_workers)

        if self.args.dump_grads_mat:
            self._all_grads = torch.stack(self._all_grads)
            dump_tensor_to_mat(self._all_grads, self.args.run_dir, 'gradsU.mat', 'U', transpose=True)
            del self._all_grads

        assert self._num_params == self._fisher_inv_diag.shape[0]
        self._param_idx = 0

        flat_pruned_weights_list = []
        flat_module_weights_list = []
        module_shapes_list = []
        module_param_indices_list = []
        prune_masks = []
        past_weight_masks = []

        #############################################################
        # Step 2. Get param stats and either jointly or independently create sparsification masks!

        # Step 2.1: If independent, then compute param stats and masks at once.
        # Else, save param stats for all modules in an array

        for idx, module in enumerate(self._modules):
            # print(f'module is {module}')
            level = self._required_sparsity(epoch_num)

            # multiplying by the current mask makes the corresponding statistic
            # of those weights zero and keeps them removed.

            past_weight_masks.append(module.weight_mask)
            module_param_indices_list.append(self._param_idx)
            assert self._weight_only
            module_shapes_list.append(module.weight.shape)

            w_stat = self._get_param_stat(module.weight, module.weight_mask, self._fisher_inv_diag, self._param_idx)
            self._param_idx += module.weight.numel()

            if self.args.woodburry_joint_sparsify:
                self._param_stats.append(w_stat.flatten())

            if module.bias is not None and not self._weight_only:
                print('sparsifying bias as well')
                b_stat = self._get_param_stat(module.bias, module.bias_mask, self._fisher_inv_diag, self._param_idx)
                self._param_idx += module.bias.numel()
                if self.args.woodburry_joint_sparsify:
                    self._param_stats.append(b_stat.flatten())

            if not self.args.woodburry_joint_sparsify:
                module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level), \
                                                       self._get_pruning_mask(None if self._weight_only else b_stat,
                                                                              level)

        # Step 2.2 For the joint sparsification case, build a global param mask
        # based on the param stats saved across various modules!
        if self.args.woodburry_joint_sparsify:
            level = self._required_sparsity(epoch_num)
            global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), level)

            _param_count = 0
            for idx, module in enumerate(self._modules):
                module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(
                    module.weight)

                _param_count += module.weight.numel()

                if module.bias is not None and not self._weight_only:
                    module.bias_mask = global_param_mask[_param_count + module.bias.numel()].view_as(module.bias)
                    _param_count += module.bias.numel()
                else:
                    module.bias_mask = None

            del self._param_stats
            del global_param_mask

        #############################################################
        # Step 3. Now that sparsification masks have been computed whether jointly or independently,
        # put them together in a list, and apply the requisite OBS update to other remaining weights

        for idx, module in enumerate(self._modules):
            assert self._weight_only
            pruned_weights = past_weight_masks[idx] - module.weight_mask
            prune_mask = past_weight_masks[idx] > module.weight_mask
            prune_masks.append(prune_mask)
            # print(f'pruned_weights are {pruned_weights}')
            pruned_weights = pruned_weights.flatten().float()
            flat_pruned_weights_list.append(pruned_weights)
            flat_module_weights_list.append(module.weight.flatten())

        module_param_indices_list.append(self._param_idx)

        flat_pruned_weights_list = flatten_tensor_list(flat_pruned_weights_list)
        flat_module_weights_list = flatten_tensor_list(flat_module_weights_list)

        # compute the weight update across all modules
        scaled_basis_vector = self._get_pruned_wts_scaled_basis(flat_pruned_weights_list, flat_module_weights_list)
        weight_updates = self._fisher_inv @ scaled_basis_vector

        if self._prune_direction:
            meta['prune_direction'] = []
            meta['original_param'] = []
            meta['mask_previous'] = []
            meta['mask_overall'] = []
            meta['quad_term'] = []

        # now apply the respective module wise weight update
        for idx, module in enumerate(self._modules):
            weight_update = weight_updates[module_param_indices_list[idx]:module_param_indices_list[idx + 1]]
            cache_weight_update_shape = weight_update.shape
            weight_update = weight_update.view_as(module.weight)

            if self._zero_after_prune:

                # This flag is used in case when analyze the loss approximation due to pruning.

                # It's purpose is to make those active in the prune_mask to be 0 weight
                # since later module.weight will be added to the weight_update.
                # because of the effect of separate OBS parameter readjustments terms in practice,
                # weight update by itself won't result in weight 0 - at the places pruned.

                # However, for most of the usage, this does not matter, as we multiply weight matrices
                # by the mask when considering pruning or retraining anyways!

                weight_update[prune_masks[idx]] = (-1 * module.weight.data[prune_masks[idx]])

            print(f'for param {idx}: norm of weight is {torch.norm(module.weight).item()}')
            print(f'for param {idx}: norm of weight update is {torch.norm(weight_update).item()}')

            if self.args.local_quadratic:
                weight_update = weight_update.view(cache_weight_update_shape)
                # (e^T F^-1 e)/2 which is what comes out,
                # when you plug in weight_update to quadratic term
                meta['quad_term'].append(torch.dot(weight_update, scaled_basis_vector) / 2)
                weight_update = weight_update.view_as(module.weight)
                print('quad term comes out to be', meta['quad_term'])

            if self._prune_direction:
                meta['prune_direction'].append(weight_update)
                meta['original_param'].append(module.weight.data.clone())
                print('idx is ', idx)
                # print(flat_pruned_weights_list)
                # dirty hack that works when only 1 layer
                meta['mask_previous'].append(
                    module.weight_mask + flat_pruned_weights_list.view(module_shapes_list[idx]).type(
                        module.weight_mask.dtype))
                meta['mask_overall'].append(module.weight_mask)

            # print('weight before is ', module.weight)
            with torch.no_grad():
                module.weight[:] = module.weight.data + weight_update
            # print('weight after is ', module.weight)
            print(f'for param {idx} after update: norm of weight is {torch.norm(module.weight).item()}')

            # print(f'weights in parameter {idx} after pruning (only for pruned) are ', module.weight[prune_masks[idx]])
            if self._prune_direction:
                print(f'weights in meta[original_param][{idx}] after pruning (only for pruned) are ',
                      meta['original_param'][idx])

        self._release_grads()

        # check if all the params whose fisher inverse was computed their value has been taken
        print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {self._fisher_inv_diag.shape[0]} \n')
        assert self._param_idx == self._fisher_inv_diag.shape[0]

        del self._fisher_inv

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic

        return True, meta
