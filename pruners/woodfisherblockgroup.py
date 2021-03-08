import torch
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats
from pruners.woodfisherblock import BlockwiseWoodburryFisherPruner
import math

class GroupBlockwiseWoodburryFisherPruner(BlockwiseWoodburryFisherPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(GroupBlockwiseWoodburryFisherPruner, self).__init__(model, inp_args, **kwargs)
        logging.info("IN GROUP BLOCK WOODBURRY")
        self._block_fisher_inv_diag = []

    def _organize_grads(self):
        # self._all_grads = torch.stack(self._all_grads)
        self.whole_block_indices = []
        whole_block_idx = 0
        while whole_block_idx < self._num_params:
            self._all_grads_dic[whole_block_idx] = self._all_grads[:, whole_block_idx: min(whole_block_idx + self.args.fittable_params, self._num_params)]
            self.whole_block_indices.append(whole_block_idx)
            whole_block_idx += min(self.args.fittable_params, self._num_params-whole_block_idx)

        assert whole_block_idx == self._num_params
        self._num_grads = self._all_grads.shape[0] # self._all_grads is a tensor of shape num_grads x num_params
        del self._all_grads
        torch.cuda.empty_cache()

    def _compute_block_woodburry_fisher_inverse(self, whole_block_idx, device):
        st_time = time.perf_counter()

        _block_fisher_inv = None
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        logging.info(f"{self._num_grads}, len (number) of grads")

        for idx in range(self._num_grads):

            sample_grads = self._all_grads_dic[whole_block_idx][idx]

            if self.args.fisher_cpu:
                sample_grads = sample_grads.cpu()

            if aux_device is not None and aux_device != torch.device('cpu'):
                sample_grads = sample_grads.to(aux_device)

            if not self.args.fisher_cpu and sample_grads.device == torch.device('cpu'):
                sample_grads = sample_grads.to(device)

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

        # whole_block_idx is the index of the first parameter of this layer
        # in other words, count of parameters before this!

        if self.args.offload_inv:
            _block_fisher_inv = _block_fisher_inv.cpu()

        self._block_fisher_inv_dic[whole_block_idx] = _block_fisher_inv

        if not hasattr(self, '_block_fisher_inv_diag') or self._block_fisher_inv_diag is None:
            self._block_fisher_inv_diag = []

        # append all the diagonals and later concatenate into a tensor
        # for the entire network
        self._block_fisher_inv_diag.append(_block_fisher_inv.diagonal())

        self._total_block_fisher_params += len(_block_fisher_inv.diagonal())

        end_time = time.perf_counter()
        logging.info("Time taken to compute one block_fisher_inverse for param idx {} with woodburry is {} seconds".format(whole_block_idx, str(end_time - st_time)))

        if self._inspect_inv:
            logging.info(f'---- Inspecting block fisher inverse for whole_block_idx {idx} ----')
            inspect_dic = get_summary_stats(_block_fisher_inv)
            inspect_dic['trace'] = _block_fisher_inv.trace().item()
            inspect_dic['sum'] = _block_fisher_inv.sum().item()
            inspect_dic['trace/sum'] = (_block_fisher_inv.trace()/_block_fisher_inv.sum()).item()
            self.inspect_dic[whole_block_idx] = inspect_dic
            logging.info(f"{self.inspect_dic[whole_block_idx]}")
            logging.info('-----------------------------------')

        del _block_fisher_inv

    def _get_weight_update(self, param_idx, scaled_basis_vector):

        num_layer_params = len(scaled_basis_vector)
        print(f"num_layer_params are ", {num_layer_params})
        scaled_basis_device = scaled_basis_vector.device

        num_params_done = 0
        weight_update = []
        whole_block_num = int(math.floor(param_idx/self.args.fittable_params))
        current_idx = param_idx
        while num_params_done < num_layer_params:
            num_params_in_block = min((whole_block_num+1)*self.args.fittable_params - current_idx, num_layer_params - num_params_done)
            whole_block_idx = whole_block_num * self.args.fittable_params
            whole_offset = current_idx - whole_block_idx
            whole_block_inv = self._block_fisher_inv_dic[whole_block_idx][whole_offset: whole_offset + num_params_in_block]

            print(f"whole_block_num {whole_block_num}, whole_block_idx {whole_block_idx}, current_idx {current_idx}, whole_offset {whole_offset}",
                  f"num_params_in_block {num_params_in_block}, num_params_done {num_params_done}")

            weight_update.append(whole_block_inv @
                                 scaled_basis_vector[num_params_done: num_params_done + num_params_in_block].to(
                                     whole_block_inv.device))

            num_params_done += num_params_in_block
            current_idx += num_params_in_block

            whole_block_num += 1

        assert num_params_done == scaled_basis_vector.shape[0]
        assert num_params_done == num_layer_params

        weight_update = torch.cat(weight_update)
        # reduce the effect of weight update
        # it is okay to not worry about the parameters that are going to be removed
        # as they will be anyways masked
        weight_update = self.args.scale_prune_update * weight_update
        if self.args.scale_prune_update < 1:
            logging.info(f"reduce the scale of pruning update by {self.args.scale_prune_update}")

        weight_update = weight_update.to(scaled_basis_device)
        return weight_update

    def organize_fisher_inv_diag(self):
        self._block_fisher_inv_diag = torch.cat(self._block_fisher_inv_diag)
        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._block_fisher_inv_diag_dic[_param_count] = self._block_fisher_inv_diag[
                                              _param_count: _param_count + module.weight.numel()]
            _param_count += module.weight.numel()

        assert _param_count == self._num_params

        del self._block_fisher_inv_diag

    def _build_weight_updates(self):
        assert self._weight_only

        self.scaled_basis_vector_list = torch.cat(self.scaled_basis_vector_list)
        print((self.scaled_basis_vector_list == 0).float().sum(), "number of zeros in self.scaled_basis_vector_list")
        scaled_basis_device = self.scaled_basis_vector_list.device
        scaled_basis_len = self.scaled_basis_vector_list.shape[0]

        self._weight_update_dic = {}
        _weight_update_list = []
        num_params_done = 0
        for whole_block_idx in self._block_fisher_inv_dic:
            print(f"whole_block_idx is {whole_block_idx}")
            whole_block_inv = self._block_fisher_inv_dic[whole_block_idx]
            block_size = whole_block_inv.shape[0]
            _weight_update_list.append(whole_block_inv @
                                   self.scaled_basis_vector_list[whole_block_idx: whole_block_idx + block_size].to(
                                     whole_block_inv.device))
            num_params_done += block_size
            del whole_block_inv
            self._block_fisher_inv_dic[whole_block_idx] = None

        del self.scaled_basis_vector_list
        assert num_params_done == scaled_basis_len
        _weight_update_list = torch.cat(_weight_update_list)

        _weight_update_list = self.args.scale_prune_update * _weight_update_list
        if self.args.scale_prune_update < 1:
            logging.info(f"reduce the scale of pruning update by {self.args.scale_prune_update}")

        _weight_update_list = _weight_update_list.to(scaled_basis_device)

        _param_count = 0
        for idx, module in enumerate(self._modules):
            self._weight_update_dic[_param_count] = _weight_update_list[_param_count:_param_count + module.weight.numel()]
            _param_count += module.weight.numel()

        assert num_params_done == _param_count

        del _weight_update_list

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            logging.info("Pruner is not ACTIVEEEE yaa!")
            return False, {}

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
        if not hasattr(self, '_block_fisher_inv_diag'):
            self._block_fisher_inv_diag = []
        if not hasattr(self, '_param_stats'):
            self._param_stats = []
        self._total_block_fisher_params = 0

        assert self.args.woodburry_joint_sparsify

        #############################################################
        # Step0: compute all grads!
        self._compute_all_grads(dset, subset_inds, device, num_workers)

        # Step0.5: organize all grads into a dic!
        self._organize_grads()
        self._release_grads()

        if self._prune_direction:
            meta['prune_direction'] = []
            meta['original_param'] = []
            meta['mask_previous'] = []
            meta['mask_overall'] = []

        self._whole_block_idx = 0
        num_whole_blocks = int(math.ceil(self._num_params/self.args.fittable_params))

        #############################################################
        # Step1: compute whole-block woodfisher inverse!
        for idx in range(num_whole_blocks):
            # print(f'module is {module}')
            self._compute_block_woodburry_fisher_inverse(self._whole_block_idx, device)
            self._all_grads_dic.pop(self._whole_block_idx, None)
            # del self._all_grads_dic[self._param_idx]
            torch.cuda.empty_cache()
            self._whole_block_idx += min(self.args.fittable_params, self._num_params - self._whole_block_idx)
        assert self._whole_block_idx == self._num_params

        self.organize_fisher_inv_diag()

        # Now, compute the pruning statistic and weight updates as usual
        self._param_idx = 0
        for idx, module in enumerate(self._modules):

            if self.args.fisher_cpu:
                w_stat = self._get_param_stat(module.weight.cpu(), module.weight_mask.cpu(),
                                              self._block_fisher_inv_diag_dic[self._param_idx], subtract_min=self.args.subtract_min)
                w_stat = w_stat.cpu()
            else:
                w_stat = self._get_param_stat(module.weight, module.weight_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx], subtract_min=self.args.subtract_min)


            # w_stat has the shape of module.weight, flatten it
            # later on after obtaining the mask unflatten!
            self._param_stats.append(w_stat.flatten())

            self._param_idx += module.weight.numel()

            if module.bias is not None and not self._weight_only:
                logging.info('sparsifying bias as well')
                self._compute_block_woodburry_fisher_inverse(module.bias, self._param_idx, device)
                b_stat = self._get_param_stat(module.bias, module.bias_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx].to(module.bias.device), subtract_min=self.args.subtract_min)
                self._param_stats.append(b_stat.flatten())
                self._param_idx += module.bias.numel()

        #############################################################
        # Step2: doing global parameter selection!
        level = self._required_sparsity(epoch_num)
        global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), level)
        logging.info(f'shape of global param mask is {list(global_param_mask.shape)}')

        # obtain masks for each layer via globalmagni to compare
        if self.args.compare_globalmagni_mask:
            self._get_magni_global_mask(epoch_num)
            if self.args.spearman_globalmagni:
                from scipy.stats import spearmanr
                coeff, p = spearmanr(flatten_tensor_list(self._param_stats).cpu().detach().numpy(),
                                     self._param_stats_magni.cpu().detach().numpy())

                logging.info(f'spearman correlation between pruning stats of global WF and magni is {coeff} with a p-value {p}')
        else:
            self._layer_masks_global_magni = None

        del self._param_stats

        #############################################################
        # Step3: computing global update!

        assert self._weight_only

        _param_count = 0
        self.scaled_basis_vector_list = []
        weight_mask_before_list = []

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
            # prune_mask = weight_mask_before > module.weight_mask # active if was 1 before and 0 now
            pruned_weights = pruned_weights.flatten().float()

            # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
            if not self.args.fisher_cpu:
                scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                        _param_count)
            else:
                scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), module.weight.flatten().cpu(),
                                                                        _param_count)

            self.scaled_basis_vector_list.append(scaled_basis_vector)
            weight_mask_before_list.append(weight_mask_before)

            self._block_fisher_inv_diag_dic[_param_count] = None
            del pruned_weights
            # del prune_mask
            del scaled_basis_vector
            del weight_mask_before

            _param_count += module.weight.numel()

            module.bias_mask = None

        assert self._param_idx == _param_count

        self._build_weight_updates()

        _param_count = 0
        # last_whole_block_num = 0
        for idx, module in enumerate(self._modules):
            weight_mask_before = weight_mask_before_list[idx]
            prune_mask = weight_mask_before > module.weight_mask # active if was 1 before and 0 now

            weight_update = self._weight_update_dic[_param_count]
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

            # print('weight before is ', module.weight)

            # print('weight after is ', module.weight)
            logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight is {torch.norm(module.weight).item()}')

            # print(f'weights in parameter {idx} named {self._module_names[idx]} after pruning (only for pruned) are ', module.weight[prune_mask])

            # inspecting if underlying references did not change
            if self._prune_direction:
                print(f'weights in meta[original_param][{idx}] after pruning (only for pruned) are ', meta['original_param'][idx])


            self._weight_update_dic[_param_count] = None

            del prune_mask
            del weight_update
            del weight_mask_before

            _param_count += module.weight.numel()

            module.bias_mask = None
            # PR: later maybe handle bias, for now set bias_mask to None
            # if module.bias is not None and not self._weight_only:
            #     module.bias_mask = global_param_mask[_param_count + module.bias.numel()].view_as(module.bias)
            #     _param_count += module.bias.numel()

        assert self._param_idx == _param_count

        # check if all the params whose fisher inverse was computed their value has been taken
        # print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {len(self._all_grads[0])} \n')
        assert self._param_idx == self._total_block_fisher_params
        assert self._num_params == self._total_block_fisher_params

        del weight_mask_before_list
        del self._block_fisher_inv_diag_dic
        del self._block_fisher_inv_dic

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic
        return True, meta