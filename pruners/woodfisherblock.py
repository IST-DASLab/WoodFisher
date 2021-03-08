import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, get_total_sparsity
from policies.pruners import GradualPruner
import math

class BlockwiseWoodburryFisherPruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(BlockwiseWoodburryFisherPruner, self).__init__(model, inp_args, **kwargs)
        logging.info("IN BLOCK WOODBURRY")
        self._all_grads = None
        self._all_grads_dic = {}
        self._block_fisher_inv_dic = {}
        self._block_fisher_inv_diag_dic = {}
        self._param_stats = []
        # need to consider the update  - ( w_q  (H^-1 e_q ) / [H^-1]_qq
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        if self._inspect_inv:
            self.inspect_dic = {}
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        self._total_block_fisher_params = 0

        if hasattr(inp_args, 'fittable_params'):
            self._fittable_params = inp_args.fittable_params
        else:
            self._fittable_params = 64 * 64 * 3

        self._previous_epoch_level = None

    def _compute_grads(self, loss):
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

        if self.args.offload_grads:
            grads = grads.cpu()

        return grads

    def _save_grads(self, loss, grads=None):
        '''
        Rememeber to filter params who are not in _module_names!
        and then also maybe maintain a way to index into the resulting diagonal computed!
        '''

        if grads is None:
            grads = self._compute_grads(loss)

        if not hasattr(self, '_all_grads') or self._all_grads is None:
            self._all_grads = grads.view(1, -1)
        else:
            self._all_grads = torch.cat([self._all_grads, grads.view(1, -1)])

        if not hasattr(self, '_num_params'):
            self._num_params = len(grads)


    def _get_pruned_wts_scaled_basis(self, pruned_params, flattened_params, param_idx):
        return -1 * torch.div(torch.mul(pruned_params, flattened_params), self._block_fisher_inv_diag_dic[param_idx].to(flattened_params.device))

    @staticmethod
    def _get_param_stat(param, param_mask, block_fisher_inv_diag, subtract_min=False, flop_stat=None):
        if param is None or param_mask is None: return None
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = block_fisher_inv_diag.view_as(param).to(param.device)
        logging.info(f"mean value of statistic without eps = {1e-10} is {(torch.mean((param ** 2)/inv_fisher_diag_entry)).item()}")
        logging.info(f"std value of statistic without eps = {1e-10} is {(torch.std((param ** 2) / inv_fisher_diag_entry)).item()}")

        # multiplying by the current mask makes the corresponding statistic
        # of those weights zero and keeps them removed.
        logging.info(f'mean value of param^2 is {(param**2).mean().item()} and std is {(param**2).std().item()}')
        logging.info(f'mean value of inv fisher is {inv_fisher_diag_entry.mean().item()} and std is {inv_fisher_diag_entry.std().item()}')
        optimal_brain_damage_stat = (param ** 2)/(inv_fisher_diag_entry + 1e-10)

        if subtract_min:
            logging.info('subtracting min in param_stat')
            optimal_brain_damage_stat = optimal_brain_damage_stat - optimal_brain_damage_stat.min()

        if flop_stat is not None:
            pruning_stat = (optimal_brain_damage_stat + 1e-10) / (flop_stat + 1e-10)
        else:
            pruning_stat = optimal_brain_damage_stat + 1e-10

        return pruning_stat * param_mask

    def _release_grads(self):
        optim.SGD(self._model.parameters(), lr=1e-10).zero_grad()

    def _add_outer_products_efficient_v1(self, mat, vec, num_parts=2):
        piece = int(math.ceil(len(vec) / num_parts))
        vec_len = len(vec)
        for i in range(num_parts):
            for j in range(num_parts):
                mat[i * piece:min((i + 1) * piece, vec_len), j * piece:min((j + 1) * piece, vec_len)].add_(
                    torch.ger(vec[i * piece:min((i + 1) * piece, vec_len)], vec[j * piece:min((j + 1) * piece, vec_len)])
                )

    def _get_loss(self, output, target, device, reduction='mean'):
        # if the criterion object was not made in the past
        if not hasattr(self, 'criterion'):
            if self.args.disable_log_soft:
                # set to true for resnet20 case
                # set to false for mlpnet as it then returns the log softmax and we go to NLL
                if not self.args.hess_label_smoothing:
                    self.criterion = torch.nn.functional.cross_entropy
                else:
                    from utils.utils import LabelSmoothing
                    self.criterion = LabelSmoothing(smoothing=self.args.hess_label_smoothing)
            else:
                self.criterion = F.nll_loss

        if self.args.true_fisher:
            sampled_y = torch.multinomial(torch.nn.functional.softmax(output.cpu().data, dim=1), 1).squeeze(1).to(
                device)
            loss = self.criterion(output, sampled_y, reduction=reduction)
        else:
            loss = self.criterion(output, target, reduction=reduction)

        return loss

    def _compute_all_grads(self, dset, subset_inds, device, num_workers, debug=False):
        st_time = time.perf_counter()
        self._model = self._model.to(device)

        logging.info("computing all grads for blockwise woodfisher: len of subset_inds is {}".format(len(subset_inds)))

        self._goal = self.args.fisher_subsample_size

        if self.args.max_mini_bsz is not None:
            assert len(subset_inds) == self._goal * int(
                math.ceil(self._fisher_mini_bsz / self.args.max_mini_bsz)) * self.args.max_mini_bsz
        else:
            assert len(subset_inds) == self._goal * self._fisher_mini_bsz
        # print("# of examples done {} and the self._goal is {}".format(num, self._goal))


        if self.args.max_mini_bsz is not None and self.args.max_mini_bsz < self._fisher_mini_bsz:
            max_mini_bsz = self.args.max_mini_bsz
        else:
            max_mini_bsz = self._fisher_mini_bsz

        dummy_loader = torch.utils.data.DataLoader(dset, batch_size=max_mini_bsz, num_workers=num_workers,
                                                   sampler=SubsetRandomSampler(subset_inds))
        logging.info(f'dummy_loader has batch size {max_mini_bsz}')
        dummy_iter = iter(dummy_loader)
        num_batches = 0
        num_samples = 0

        while num_samples != (self._goal * self._fisher_mini_bsz):
            self._release_grads()

            # print(f"fisher_mini_bsz is {self._fisher_mini_bsz} and max_mini_bsz is {max_mini_bsz}")
            if max_mini_bsz < self._fisher_mini_bsz:
                # logging.info(f'processing in smaller batches of size `max_mini_bsz`: {max_mini_bsz} for fisher_mini_bsz={self._fisher_mini_bsz}')

                mini_batch_seen = 0

                while mini_batch_seen < self._fisher_mini_bsz:
                    mini_in_tensor, mini_target = next(dummy_iter)
                    if mini_in_tensor.shape[0] > (self._fisher_mini_bsz - mini_batch_seen):
                        # if the number of samples obtained is more than necessary
                        mini_in_tensor = mini_in_tensor[0:self._fisher_mini_bsz - mini_batch_seen]
                        mini_target = mini_target[0:self._fisher_mini_bsz - mini_batch_seen]

                    mini_in_tensor, mini_target = mini_in_tensor.to(device), mini_target.to(device)
                    mini_output = self._model(mini_in_tensor)

                    loss = self._get_loss(mini_output, mini_target, device, reduction='sum')

                    if mini_batch_seen == 0:
                        intmd_grads = self._compute_grads(loss)
                    else:
                        intmd_grads += self._compute_grads(loss)

                    mini_batch_seen += mini_in_tensor.shape[0]

                    del mini_in_tensor
                    del mini_target
                    del mini_output
                    torch.cuda.empty_cache()

                assert mini_batch_seen == self._fisher_mini_bsz

                intmd_grads /= mini_batch_seen

                self._save_grads(loss=None, grads=intmd_grads)

            else:
                in_tensor, target = next(dummy_iter)
                in_tensor, target = in_tensor.to(device), target.to(device)
                output = self._model(in_tensor)
                loss = self._get_loss(output, target, device, reduction='mean')

                # The default reduction = 'mean' will be used, over the fisher_mini_bsz,
                # which is just a practical heuristic to utilize more datapoints

                self._save_grads(loss)

            num_batches += 1
            num_samples += self._fisher_mini_bsz
            # print("# of examples done {} and the self._goal is {}".format(num, self._goal))

            # if num_samples == (self._goal * self._fisher_mini_bsz):
            #     break

        assert num_samples == (self._goal * self._fisher_mini_bsz)
        logging.info("# of examples done {} and the self._goal is {}".format(num_samples, self._goal))
        logging.info("# of batches done {}".format(num_batches))

        end_time = time.perf_counter()
        logging.info("Time taken to save all grads is {} seconds".format(str(end_time - st_time)))


    def _organize_grads(self):

        if self.args.dump_grads_mat:
            import scipy.io
            import os
            scipy.io.savemat(os.path.join(self.args.run_dir, 'gradsU.mat'),
                             dict(U=self._all_grads.t().cpu().numpy()))

        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._all_grads_dic[_param_count] = self._all_grads[:, _param_count:_param_count + module.weight.numel()]
            _param_count += module.weight.numel()

        assert _param_count == self._num_params
        self._num_grads = self._all_grads.shape[0]
        del self._all_grads
        torch.cuda.empty_cache()

    def _compute_split_block_woodburry_fisher_inverse(self, param_idx, start, end, device):

        st_time = time.perf_counter()

        _sub_block_fisher_inv = None
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')
        logging.info(f'In sub_block for param_idx {param_idx}, with split indices: start {start} and end {end}')
        logging.info(f"{self._num_grads}, len (number) of grads")

        for idx in range(self._num_grads):

            # sample_grads = self._all_grads[idx][param_idx: param_idx + param.numel()]
            sample_grads = self._all_grads_dic[param_idx][idx][start:end]

            if self.args.fisher_cpu:
                sample_grads = sample_grads.cpu()

            if aux_device is not None and aux_device != torch.device('cpu'):
                sample_grads = sample_grads.to(aux_device)

            if not self.args.fisher_cpu and sample_grads.device == torch.device('cpu'):
                sample_grads = sample_grads.to(device)

            # print(f'device of sample_grads is {sample_grads.device}')
            if idx == 0:
                # rewrite in terms of inplace operations

                numerator_normalization = (self.args.fisher_damp) ** 2

                _sub_block_fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                    self._goal + (sample_grads.dot(sample_grads) / self.args.fisher_damp)
                )
                _sub_block_fisher_inv.diagonal().sub_(1.0 / self.args.fisher_damp)
                # 1/self.args.fisher_damp \times Identity matrix is used to represent (H^-1)_0
                _sub_block_fisher_inv.mul_(-1)

            else:
                cache_matmul = torch.matmul(_sub_block_fisher_inv, sample_grads)
                cache_matmul.div_((self._goal + sample_grads.dot(cache_matmul)) ** 0.5)

                if not self.args.fisher_optimized:
                    _sub_block_fisher_inv.sub_(
                        torch.ger(cache_matmul, cache_matmul)
                    )
                else:
                    assert self.args.fisher_parts > 1
                    # F = F - x x'
                    # F1 = -F
                    _sub_block_fisher_inv.mul_(-1)
                    # F1 + x x'
                    self._add_outer_products_efficient_v1(
                        _sub_block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts
                    )
                    # F = - F1
                    _sub_block_fisher_inv.mul_(-1)

                del cache_matmul

                del sample_grads

        end_time = time.perf_counter()
        logging.info("Time taken to compute the sub block_fisher_inverse for param idx {} with woodburry is {} seconds".format(
            param_idx, str(end_time - st_time)))

        if self.args.offload_inv:
            _sub_block_fisher_inv = _sub_block_fisher_inv.cpu()

        return _sub_block_fisher_inv

    def _compute_block_woodburry_fisher_inverse(self, param, param_idx, device):
        st_time = time.perf_counter()

        _block_fisher_inv = None
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        logging.info(f"{self._num_grads}, len (number) of grads")

        if not self.args.fisher_split_grads or param.numel() <= self._fittable_params:
            for idx in range(self._num_grads):

                # sample_grads = self._all_grads[idx][param_idx: param_idx + param.numel()]
                sample_grads = self._all_grads_dic[param_idx][idx]

                if self.args.fisher_cpu:
                    sample_grads = sample_grads.cpu()

                if aux_device is not None and aux_device != torch.device('cpu'):
                    sample_grads = sample_grads.to(aux_device)

                if not self.args.fisher_cpu and sample_grads.device == torch.device('cpu'):
                    sample_grads = sample_grads.to(device)

                # print(f'device of sample_grads is {sample_grads.device}')
                if idx == 0:
                    # rewrite in terms of inplace operations

                    numerator_normalization = (self.args.fisher_damp) ** 2

                    _block_fisher_inv = torch.ger(sample_grads, sample_grads).mul_(1.0 / numerator_normalization).div_(
                        self._goal + (sample_grads.dot(sample_grads) / self.args.fisher_damp)
                    )
                    _block_fisher_inv.diagonal().sub_(1.0 / self.args.fisher_damp)
                    # 1/self.args.fisher_damp \times Identity matrix is used to represent (H^-1)_0
                    _block_fisher_inv.mul_(-1)

                else:
                    cache_matmul = torch.matmul(_block_fisher_inv, sample_grads)
                    cache_matmul.div_((self._goal + sample_grads.dot(cache_matmul)) ** 0.5)

                    if not self.args.fisher_optimized:
                        _block_fisher_inv.sub_(
                            torch.ger(cache_matmul, cache_matmul)
                        )
                    else:
                        assert self.args.fisher_parts > 1
                        # F = F - x x'
                        # F1 = -F
                        _block_fisher_inv.mul_(-1)
                        # F1 + x x'
                        self._add_outer_products_efficient_v1(
                            _block_fisher_inv, cache_matmul, num_parts=self.args.fisher_parts
                        )
                        #F = - F1
                        _block_fisher_inv.mul_(-1)

                    del cache_matmul

                    del sample_grads
            # param_idx is the index of the first parameter of this layer
            # in other words, count of parameters before this!

            if self.args.offload_inv:
                _block_fisher_inv = _block_fisher_inv.cpu()

            self._block_fisher_inv_dic[param_idx] = _block_fisher_inv
            self._block_fisher_inv_diag_dic[param_idx] = _block_fisher_inv.diagonal()

        else:
            # if the number of params is > fittable parameter limit
            num_params = param.numel()
            num_splits = int(math.ceil(num_params/self._fittable_params))
            _block_fisher_inv = []
            _block_fisher_inv_diag = []
            split_start = 0
            for split_idx in range(num_splits):
                split_end = min((split_idx + 1) * self._fittable_params, num_params)
                _block_fisher_inv.append(self._compute_split_block_woodburry_fisher_inverse(param_idx, split_start, split_end, device))
                _block_fisher_inv_diag.append(_block_fisher_inv[-1].diagonal())
                split_start += min(self._fittable_params, num_params-split_start)
            assert split_start == num_params
            _block_fisher_inv_diag = torch.cat(_block_fisher_inv_diag)

            self._block_fisher_inv_diag_dic[param_idx] = _block_fisher_inv_diag
            self._block_fisher_inv_dic[param_idx] = _block_fisher_inv

        self._total_block_fisher_params += len(self._block_fisher_inv_diag_dic[param_idx])

        end_time = time.perf_counter()
        logging.info("Time taken to compute one block_fisher_inverse for param idx {} with woodburry is {} seconds".format(param_idx, str(end_time - st_time)))

        if self._inspect_inv:
            logging.info(f'---- Inspecting block fisher inverse for param_idx {idx} ----')
            inspect_dic = get_summary_stats(self._block_fisher_inv)
            inspect_dic['trace'] = self._block_fisher_inv.trace().item()
            inspect_dic['sum'] = self._block_fisher_inv.sum().item()
            inspect_dic['trace/sum'] = (self._block_fisher_inv.trace()/self._block_fisher_inv.sum()).item()
            self.inspect_dic[param_idx] = inspect_dic
            logging.info(f"{self.inspect_dic[param_idx]}")
            logging.info('-----------------------------------')

    def _compute_single_layer_trace(self, grads):
        logging.info(f"shape of grads is {list(grads.shape)}")
        return (torch.norm(grads, dim=1)**2).sum()/grads.shape[0]

    def _compute_layerwise_fisher_traces(self):
        self._trace_dic = {}
        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._trace_dic[self._module_names[idx]] = self._compute_single_layer_trace(self._all_grads_dic[_param_count])
            _param_count += module.weight.numel()
        assert _param_count == self._num_params
        logging.info(f"This is the dictionary containing layerwise traces{self._trace_dic}")

    def _get_weight_update(self, param_idx, scaled_basis_vector):
        scaled_basis_device = scaled_basis_vector.device

        if not self.args.fisher_split_grads or type(self._block_fisher_inv_dic[param_idx])!=list:
            weight_update = self._block_fisher_inv_dic[param_idx] @ scaled_basis_vector.to(self._block_fisher_inv_dic[param_idx].device)
        else:
            weight_update = []
            split_start = 0
            # print(f"shape of scaled_basis vector at param_idx {param_idx} is", scaled_basis_vector.shape)
            for split_idx in range(len(self._block_fisher_inv_dic[param_idx])):
                # print(f"shape of subblock at split_idx {split_idx} is ", self._block_fisher_inv_dic[param_idx][split_idx].shape)
                split_len = self._block_fisher_inv_dic[param_idx][split_idx].shape[0]
                weight_update.append(self._block_fisher_inv_dic[param_idx][split_idx] @
                                     scaled_basis_vector[split_start:split_start + split_len].to(self._block_fisher_inv_dic[param_idx][split_idx].device))
                split_start += split_len

            assert split_start == scaled_basis_vector.shape[0]
            weight_update = torch.cat(weight_update)

        # "scale_prune_update": reduces the effect of weight update.
        # Also, it is okay to not worry about the parameters that are going to be removed
        # as they will be anyways masked.
        weight_update = self.args.scale_prune_update * weight_update
        if self.args.scale_prune_update < 1:
            logging.info(f"reduce the scale of pruning update by {self.args.scale_prune_update}")

        weight_update = weight_update.to(scaled_basis_device)
        return weight_update

    @staticmethod
    def _get_param_stat_magni(param, param_mask):
        if param is None or param_mask is None: return None
        # magnitude pruner with stat as magnitude
        return (param.abs() + 1e-4) * param_mask

    def _get_magni_global_mask(self, epoch_num):
        # method to analyze global magni masks with WF based
        logging.info("Getting global magni mask. Works ONLY for one-shot case!")
        _param_stats_magni = []

        assert self._weight_only

        for module in self._modules:
            # Will have to change the module.weight_mask to that via global magni if
            # you want to compare the masks beyond one-shot case
            w_stat, b_stat = self._get_param_stat_magni(module.weight, module.weight_mask),\
                             self._get_param_stat_magni(module.bias, module.bias_mask)

            _param_stats_magni.append(w_stat.flatten())

        level = self._required_sparsity(epoch_num)

        # decide which to prune globally
        global_param_mask_magni = self._get_pruning_mask(flatten_tensor_list(_param_stats_magni), level)

        if self.args.spearman_globalmagni:
            logging.info('saving globalmagni pruning stats')
            self._param_stats_magni = flatten_tensor_list(_param_stats_magni)
        else:
            logging.info('setting globalmagni pruning stats to None')
            self._param_stats_magni = None
            del _param_stats_magni

        self._layer_masks_global_magni = {}
        # obtain the layerwise masks from this global mask
        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._layer_masks_global_magni[idx] = global_param_mask_magni[_param_count:_param_count + module.weight.numel()].view_as(
                module.weight)
            _param_count += module.weight.numel()

        del global_param_mask_magni

    def prune_to_level(self, desired_level, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        # ensure that the model is not in training mode, this is importance, because
        # otherwise the pruning procedure will interfere and affect the batch-norm statistics
        assert not self._model.training

        # reinit params if were deleted during gradual pruning
        if not hasattr(self, '_all_grads'):
            self._all_grads = None
        if not hasattr(self, '_all_grads_dic'):
            self._all_grads_dic = {}
        if not hasattr(self, '_block_fisher_inv_dic'):
            self._block_fisher_inv_dic = {}
        if not hasattr(self, '_block_fisher_inv_diag_dic'):
            self._block_fisher_inv_diag_dic = {}
        if not hasattr(self, '_param_stats'):
            self._param_stats = []
        self._total_block_fisher_params = 0

        #############################################################
        # Step0: compute all grads!
        self._compute_all_grads(dset, subset_inds, device, num_workers)

        # Step0.5: organize all grads into a dic!
        self._organize_grads()
        self._release_grads()

        if self.args.fisher_trace:
            self._compute_layerwise_fisher_traces()

        if self._prune_direction:
            meta['prune_direction'] = []
            meta['original_param'] = []
            meta['mask_previous'] = []
            meta['mask_overall'] = []

        if self.args.flops_power > 0:
            assert self.args.woodburry_joint_sparsify
            flop_stat_dic = self._get_flop_stats()

        self._param_idx = 0

        #############################################################
        # Step1: compute blockwise woodfisher inverse!

        for idx, module in enumerate(self._modules):
            self._compute_block_woodburry_fisher_inverse(module.weight, self._param_idx, device)
            self._all_grads_dic.pop(self._param_idx, None)
            torch.cuda.empty_cache()

            if self.args.flops_power > 0 and flop_stat_dic is not None:
                flop_stat = flop_stat_dic[self._module_names[idx]]
            else:
                flop_stat = None

            if self.args.fisher_cpu:
                w_stat = self._get_param_stat(module.weight.cpu(), module.weight_mask.cpu(),
                                              self._block_fisher_inv_diag_dic[self._param_idx],
                                              subtract_min=self.args.subtract_min, flop_stat=flop_stat)
            else:
                w_stat = self._get_param_stat(module.weight, module.weight_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx],
                                              subtract_min=self.args.subtract_min, flop_stat=flop_stat)

            if not self.args.woodburry_joint_sparsify:
                assert self._weight_only
                weight_mask_before = module.weight_mask
                module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, desired_level), None

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                prune_mask = weight_mask_before > module.weight_mask  # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])

                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                            self._param_idx)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), module.weight.flatten().cpu(),
                                                                            self._param_idx)

                # weight_update = self._block_fisher_inv_dic[self._param_idx].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = self._get_weight_update(self._param_idx, scaled_basis_vector)
                weight_update = weight_update.view_as(module.weight).to(module.weight.device)
                logging.info(f'at idx {idx} named {self._module_names[idx]}, shape of weight update is {list(weight_update.shape)}')

                if self._zero_after_prune:

                    # This flag is used in case when analyze the loss approximation due to pruning.

                    # It's purpose is to make those active in the prune_mask to be 0 weight
                    # since later module.weight will be added to the weight_update.
                    # because of the effect of separate OBS parameter readjustments terms in practice,
                    # weight update by itself won't result in weight 0 - at the places pruned.

                    # However, for most of the usage, this does not matter, as we multiply weight matrices
                    # by the mask when considering pruning or retraining anyways!

                    weight_update[prune_mask] = (-1 * module.weight.data[prune_mask])

                if self.args.fisher_cpu:
                    weight_update = weight_update.to(module.weight.device)
                    module.weight_mask = module.weight_mask.to(module.weight.device)
                    if module.bias_mask is not None:
                        module.bias_mask = module.bias_mask.to(module.weight.device)

                    print('device of weight_update is ', weight_update.device)
                    print('device of module.weight_mask is ', module.weight_mask.device)

                logging.info(f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight is {torch.norm(module.weight).item()}')
                logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight update is {torch.norm(weight_update).item()}')

                if self._prune_direction:
                    meta['prune_direction'].append(weight_update)
                    meta['original_param'].append(module.weight.data.clone())
                    meta['mask_previous'].append(weight_mask_before)
                    meta['mask_overall'].append(module.weight_mask)

                # save memory when doing joint sparsification!
                del w_stat
                del self._block_fisher_inv_diag_dic[self._param_idx]
                del self._block_fisher_inv_dic[self._param_idx]
                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before
            else:
                # w_stat has the shape of module.weight, flatten it
                # later on after obtaining the mask unflatten!
                if self.args.fisher_cpu:
                    w_stat = w_stat.cpu()

                self._param_stats.append(w_stat.flatten())

            self._param_idx += module.weight.numel()

            if module.bias is not None and not self._weight_only:
                logging.info('sparsifying bias as well')
                self._compute_block_woodburry_fisher_inverse(module.bias, self._param_idx, device)
                assert self.args.flops_power <= 0 # PR: flop_based statistic not implemented for bias yet
                b_stat = self._get_param_stat(module.bias, module.bias_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx].to(module.bias.device), subtract_min=self.args.subtract_min)
                self._param_stats.append(b_stat.flatten())
                self._param_idx += module.bias.numel()


        if self.args.woodburry_joint_sparsify:

            #############################################################
            # Step2: doing global parameter selection!


            global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), desired_level)
            logging.info(f'shape of global param mask is {list(global_param_mask.shape)}')

            # obtain masks for each layer via globalmagni to compare
            if self.args.compare_globalmagni_mask:
                self._get_magni_global_mask(epoch_num)
                if self.args.spearman_globalmagni:
                    from scipy.stats import spearmanr
                    coeff, p = spearmanr(flatten_tensor_list(self._param_stats).cpu().detach().numpy(),
                                         self._param_stats_magni.cpu().detach().numpy())
                    # coeff, p = spearmanr(np.sort(flatten_tensor_list(self._param_stats).cpu().detach().numpy()),
                    #                      np.sort(self._param_stats_magni.cpu().detach().numpy()))

                    logging.info(f'spearman correlation between pruning stats of global WF and magni is {coeff} with a p-value {p}')
            else:
                self._layer_masks_global_magni = None

            del self._param_stats

            #############################################################
            # Step3: computing global update!

            assert self._weight_only

            _param_count = 0
            for idx, module in enumerate(self._modules):
                weight_mask_before = module.weight_mask
                module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(module.weight)

                if self.args.compare_globalmagni_mask:
                    mask_dot_prod = (module.weight_mask * self._layer_masks_global_magni[idx]).float().sum()
                    mask_dot_prod_normalized = mask_dot_prod/(module.weight_mask.float().sum())
                    logging.info(f'at idx {idx} named {self._module_names[idx]}, the dot product of Global WF and magni masks is {mask_dot_prod} and when normalized with respect to active params is {mask_dot_prod_normalized}')

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                prune_mask = weight_mask_before > module.weight_mask # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                            _param_count)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), module.weight.flatten().cpu(),
                                                                            _param_count)

                # weight_update = self._block_fisher_inv_dic[_param_count].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = self._get_weight_update(_param_count, scaled_basis_vector)
                weight_update = weight_update.view_as(module.weight)
                logging.info(f'at idx {idx} named {self._module_names[idx]}, shape of weight update is {list(weight_update.shape)}')

                if self._zero_after_prune:

                    # This flag is used in case when analyze the loss approximation due to pruning.

                    # It's purpose is to make those active in the prune_mask to be 0 weight
                    # since later module.weight will be added to the weight_update.
                    # because of the effect of separate OBS parameter readjustments terms in practice,
                    # weight update by itself won't result in weight 0 - at the places pruned.

                    # However, for most of the usage, this does not matter, as we multiply weight matrices
                    # by the mask when considering pruning or retraining anyways!

                    weight_update[prune_mask] = (-1 * module.weight.data[prune_mask])

                if self.args.fisher_cpu:
                    weight_update = weight_update.to(module.weight.device)
                    module.weight_mask = module.weight_mask.to(module.weight.device)
                    if module.bias_mask is not None:
                        module.bias_mask = module.bias_mask.to(module.weight.device)

                    print('device of weight_update is ', weight_update.device)
                    print('device of module.weight_mask is ', module.weight_mask.device)

                logging.info(f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight update is {torch.norm(weight_update).item()}')

                if self._prune_direction:
                    meta['prune_direction'].append(weight_update)
                    meta['original_param'].append(module.weight.data.clone())
                    meta['mask_previous'].append(weight_mask_before)
                    meta['mask_overall'].append(module.weight_mask)


                logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight is {torch.norm(module.weight).item()}')

                # print(f'weights in parameter {idx} named {self._module_names[idx]} after pruning (only for pruned) are ', module.weight[prune_mask])

                # inspecting if underlying references did not change
                if self._prune_direction:
                    print(f'weights in meta[original_param][{idx}] after pruning (only for pruned) are ', meta['original_param'][idx])

                self._block_fisher_inv_dic[_param_count] = None
                self._block_fisher_inv_diag_dic[_param_count] = None

                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before

                _param_count += module.weight.numel()

                module.bias_mask = None
                # PR: Handle bias with something like the following, for now set bias_mask to None
                # if module.bias is not None and not self._weight_only:
                #     module.bias_mask = global_param_mask[_param_count + module.bias.numel()].view_as(module.bias)
                #     _param_count += module.bias.numel()


            assert self._param_idx == _param_count


        # check if all the params whose fisher inverse was computed their value has been taken
        # print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {len(self._all_grads[0])} \n')
        assert self._param_idx == self._total_block_fisher_params
        assert self._num_params == self._total_block_fisher_params

        del self._block_fisher_inv_diag_dic
        del self._block_fisher_inv_dic

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic
        return True, meta

    def _get_sparsity_info(self, idx=None):
        sparsity_dict = {}
        if idx is not None:
            logging.info(f"For recompute idx {idx}")
        for _name, _module in zip(self._module_names, self._modules):
            num_zeros, num_params = get_total_sparsity(_module)
            logging.info(f'layer named {_name} has {num_zeros} zeros out of total {num_params} params, % is {(num_zeros*1.0)/num_params}')
            sparsity_dict[_name] = (num_zeros, num_params)
        print("this is sparsity_dict", sparsity_dict)
        del sparsity_dict

    def _update_meta(self, recompute_idx, global_meta, meta):
        if recompute_idx == 0:
            global_meta = meta
        else:
            # mask previous and original_param remain the same
            global_meta['mask_overall'] = meta['mask_overall']

            # the weight updates get added !
            for idx, weight_update in enumerate(meta['prune_direction']):
                global_meta['prune_direction'][idx] += weight_update
        return global_meta

    def _recompute_level(self, recompute_idx, initial_sparsity, target_sparsity):

        start = 0
        end = self.args.recompute_num

        if self.args.recompute_schedule == 'linear':
            degree = 1
        elif self.args.recompute_schedule == 'poly':
            degree = self.args.recompute_degree

        scale = target_sparsity - initial_sparsity
        progress = min(float(recompute_idx - start + 1) / (end - start + 1), 1.0)
        remaining_progress = (1.0 - progress) ** degree

        returned_sparsity = scale * (1 - remaining_progress)

        returned_sparsity += initial_sparsity

        if recompute_idx == self.args.recompute_num:
            # ensure that the final prune level is the same
            assert returned_sparsity == target_sparsity

        return returned_sparsity

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        global_meta = {}

        if self._pruner_not_active(epoch_num):
            logging.info("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        if self._prune_direction:
            global_meta['prune_direction'] = []
            global_meta['original_param'] = []
            global_meta['mask_previous'] = []
            global_meta['mask_overall'] = []

        epoch_level = self._required_sparsity(epoch_num)

        if self._previous_epoch_level is None:
            self._previous_epoch_level = 0.0


        if self.args.recompute_num and self.args.recompute_num > 0:
            # prune in small recompute_num + 1 times
            for recompute_idx in range(0, self.args.recompute_num + 1):

                recompute_level = self._recompute_level(recompute_idx, self._previous_epoch_level, epoch_level)
                self._get_sparsity_info(idx=recompute_idx)
                print(f"At recompute idx {recompute_idx}, previous sparsity: {self._previous_epoch_level} and after: {recompute_level}")
                flag, meta = self.prune_to_level(recompute_level, dset, subset_inds, device, num_workers, epoch_num, **kwargs)
                if self._prune_direction:
                    global_meta = self._update_meta(recompute_idx, global_meta, meta)

                self._get_sparsity_info(idx=recompute_idx)
        else:
            flag, global_meta = self.prune_to_level(epoch_level, dset, subset_inds, device, num_workers, epoch_num, **kwargs)

        self._previous_epoch_level = epoch_level

        return flag, global_meta