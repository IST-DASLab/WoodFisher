import torch
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats
from pruners.woodfisherblock import BlockwiseWoodburryFisherPruner
import math

class DynamicBlockwiseWoodburryFisherPruner(BlockwiseWoodburryFisherPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(DynamicBlockwiseWoodburryFisherPruner, self).__init__(model, inp_args, **kwargs)
        logging.info("IN DYNAMIC BLOCK WOODBURRY")
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

        self._fittable_params_history = []

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

        active_grads = []

        num_active = 0
        for idx, module in enumerate(self._modules):
            active_grads.append(grads[idx][module.weight_mask > 0])
            num_selected = int(module.weight_mask.float().sum())
            num_active += num_selected
            assert len(active_grads[-1]) == num_selected

        active_grads = flatten_tensor_list(active_grads)

        # print(f"len of grads is {len(grads)} and that of active grads is {len(active_grads)}")
        assert len(active_grads) == num_active

        if self.args.offload_grads:
            active_grads = active_grads.cpu()

        del grads

        return active_grads

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

        self._num_active_params = len(grads)

    @staticmethod
    def _get_param_stat(param, param_mask, block_fisher_inv_diag, subtract_min=False):
        if param is None or param_mask is None: return None

        # masking the params by the current mask
        # w_i **2 x ((F)^-1)_ii,
        inv_fisher_diag_entry = block_fisher_inv_diag.view_as(param[param_mask > 0]).to(param.device)
        logging.info(f"mean value of statistic without eps = {1e-10} is {(torch.mean((param[param_mask > 0] ** 2)/inv_fisher_diag_entry)).item()}")
        logging.info(f"std value of statistic without eps = {1e-10} is {(torch.std((param[param_mask > 0] ** 2) / inv_fisher_diag_entry)).item()}")

        # multiplying by the current mask makes the corresponding statistic
        # of those weights zero and keeps them removed.
        logging.info(f'mean value of param^2 is {(param[param_mask > 0]**2).mean().item()} and std is {(param[param_mask > 0]**2).std().item()}')
        logging.info(f'mean value of inv fisher is {inv_fisher_diag_entry.mean().item()} and std is {inv_fisher_diag_entry.std().item()}')

        active_stat = (param[param_mask > 0] ** 2)/(inv_fisher_diag_entry + 1e-10)
        stat = torch.zeros_like(param)
        stat[param_mask > 0] = active_stat

        if subtract_min:
            logging.info('subtracting min in param_stat')
            stat = stat - stat.min()

        return (stat + 1e-10)

    def _organize_grads(self):

        if self.args.dump_grads_mat:
            import scipy.io
            import os
            scipy.io.savemat(os.path.join(self.args.run_dir, 'gradsU.mat'),
                             dict(U=self._all_grads.t().cpu().numpy()))

        _param_count = 0

        for idx, module in enumerate(self._modules):
            num_active_weights = int(module.weight_mask.float().sum())
            self._all_grads_dic[_param_count] = self._all_grads[:, _param_count:_param_count + num_active_weights]
            _param_count += num_active_weights

        print(f"Param count is {_param_count} and num_active_param is {self._num_active_params}")
        assert _param_count == self._num_active_params
        self._num_grads = self._all_grads.shape[0]
        del self._all_grads
        torch.cuda.empty_cache()

    def _compute_block_woodburry_fisher_inverse(self, num_active_param, param_idx, device):

        st_time = time.perf_counter()

        _block_fisher_inv = None
        if self.args.aux_gpu_id != -1:
            aux_device = torch.device('cuda:{}'.format(self.args.aux_gpu_id))
        else:
            aux_device = torch.device('cpu')

        logging.info(f"{self._num_grads}, len (number) of grads")

        if not self.args.fisher_split_grads or num_active_param <= self._fittable_params:
            for idx in range(self._num_grads):

                sample_grads = self._all_grads_dic[param_idx][idx]

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

            # param_idx is the index of the first parameter of this layer
            # in other words, count of parameters before this!

            if self.args.offload_inv:
                _block_fisher_inv = _block_fisher_inv.cpu()

            self._block_fisher_inv_dic[param_idx] = _block_fisher_inv
            self._block_fisher_inv_diag_dic[param_idx] = _block_fisher_inv.diagonal()

        else:
            # if the number of params is > fittable parameter limit
            num_params = num_active_param
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

    @staticmethod
    def num_active_params(module):
        return int(module.weight_mask.float().sum())

    def on_epoch_begin(self, dset, subset_inds, device, num_workers, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            logging.info("Pruner is not ACTIVEEEE yaa!")
            return False, {}

        self._fittable_params_history.append(self._fittable_params)

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

        if self._prune_direction:
            meta['prune_direction'] = []
            meta['original_param'] = []
            meta['mask_previous'] = []
            meta['mask_overall'] = []

        self._param_idx = 0
        _future_active_param_count = 0
        _real_param_count = 0
        _active_param_count = 0
        self._active_param_idx = []

        #############################################################
        # Step1: compute blockwise wood fisher inverse!
        for idx, module in enumerate(self._modules):

            active_params_before_mask_update = self.num_active_params(module)

            self._active_param_idx.append(self._param_idx)
            self._compute_block_woodburry_fisher_inverse(self.num_active_params(module), self._param_idx, device)
            self._all_grads_dic.pop(self._param_idx, None)
            torch.cuda.empty_cache()

            if self.args.fisher_cpu:
                w_stat = self._get_param_stat(module.weight.cpu(), module.weight_mask.cpu(),
                                              self._block_fisher_inv_diag_dic[self._param_idx], subtract_min=self.args.subtract_min)
            else:
                w_stat = self._get_param_stat(module.weight, module.weight_mask,
                                              self._block_fisher_inv_diag_dic[self._param_idx], subtract_min=self.args.subtract_min)

            if not self.args.woodburry_joint_sparsify:
                print("WARNING: The independent version of the pruner is not completely tested for woodfisherblockdynamic")

                assert self._weight_only
                weight_mask_before = module.weight_mask

                level = self._required_sparsity(epoch_num)
                module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level), None

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                pruned_weights = pruned_weights[weight_mask_before > 0] # this still needs to be `activated`
                prune_mask = weight_mask_before > module.weight_mask  # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])

                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, (module.weight[weight_mask_before > 0]).flatten(),
                                                                            self._param_idx)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), (module.weight[weight_mask_before > 0]).flatten().cpu(),
                                                                            self._param_idx)

                # weight_update = self._block_fisher_inv_dic[self._param_idx].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = torch.zeros_like(module.weight).flatten()
                weight_update[weight_mask_before.flatten() > 0] = self._get_weight_update(self._param_idx, scaled_basis_vector)
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

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight is {torch.norm(module.weight).item()}')
                logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight update is {torch.norm(weight_update).item()}')

                if self._prune_direction:
                    meta['prune_direction'].append(weight_update)
                    meta['original_param'].append(module.weight.data.clone())
                    meta['mask_previous'].append(weight_mask_before)
                    meta['mask_overall'].append(module.weight_mask)

                _future_active_param_count += self.num_active_params(module)
                _real_param_count += module.weight.numel()
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

            self._param_idx += active_params_before_mask_update

            assert self._weight_only

        if self.args.woodburry_joint_sparsify:
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

            _future_active_param_count = 0
            _active_param_count = 0
            _real_param_count = 0
            for idx, module in enumerate(self._modules):
                weight_mask_before = module.weight_mask
                active_params_before_mask_update = self.num_active_params(module)
                module.weight_mask = global_param_mask[_real_param_count:_real_param_count + module.weight.numel()].view_as(module.weight)

                if self.args.compare_globalmagni_mask:
                    mask_dot_prod = (module.weight_mask * self._layer_masks_global_magni[idx]).float().sum()
                    mask_dot_prod_normalized = mask_dot_prod/(module.weight_mask.float().sum())
                    logging.info(f'at idx {idx} named {self._module_names[idx]}, the dot product of Global WF and magni masks is {mask_dot_prod} and when normalized with respect to active params is {mask_dot_prod_normalized}')

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                pruned_weights = pruned_weights[weight_mask_before > 0]  # this still needs to be `activated`
                prune_mask = weight_mask_before > module.weight_mask # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, (module.weight[weight_mask_before > 0]).flatten(),
                                                                            _active_param_count)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(), (module.weight[weight_mask_before > 0]).flatten().cpu(),
                                                                            _active_param_count)

                # weight_update = self._block_fisher_inv_dic[_active_param_count].to(scaled_basis_vector.device) @ scaled_basis_vector
                weight_update = torch.zeros_like(module.weight).flatten()
                weight_update[weight_mask_before.flatten() > 0] = self._get_weight_update(_active_param_count, scaled_basis_vector)
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

                self._block_fisher_inv_dic[_active_param_count] = None
                self._block_fisher_inv_diag_dic[_active_param_count] = None

                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before

                _active_param_count += active_params_before_mask_update
                _real_param_count += module.weight.numel()
                _future_active_param_count += self.num_active_params(module)

                module.bias_mask = None
                # PR: later maybe handle bias, for now set bias_mask to None
                # if module.bias is not None and not self._weight_only:
                #     module.bias_mask = global_param_mask[_active_param_count + module.bias.numel()].view_as(module.bias)
                #     _param_count += module.bias.numel()


            assert self._param_idx == _active_param_count

        if not self.args.woodburry_joint_sparsify:
            _active_param_count = self._param_idx

        logging.info(f"Epoch[{epoch_num}]:: Active param count was {_active_param_count}, and next time will be {_future_active_param_count}, while Real param count is {_real_param_count}")

        self._fittable_params = min(int(math.floor(self.args.fittable_params * _real_param_count/float(_future_active_param_count))), _future_active_param_count)

        # assert self._fittable_params <= _active_param_count
        logging.info(f"Epoch[{epoch_num}]:: Fittable params before was {self._fittable_params_history[-1]} and next time will be {self._fittable_params}")

        # check if all the params whose fisher inverse was computed their value has been taken
        # print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {len(self._all_grads[0])} \n')
        assert self._param_idx == self._total_block_fisher_params
        assert self._num_active_params == self._total_block_fisher_params

        del self._block_fisher_inv_diag_dic
        del self._block_fisher_inv_dic

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic
        return True, meta
