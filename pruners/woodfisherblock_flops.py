import torch
import time
import logging
from utils.utils import flatten_tensor_list, get_summary_stats, get_total_sparsity
from pruners.woodfisherblock import BlockwiseWoodburryFisherPruner
from utils.flop_utils import get_macs_dpf
import math
import copy
from utils.checkpoints import get_unwrapped_model
from utils.pruner_utils import pruner_after_parameter_optimization_functional

class FlopsBlockwiseWoodburryFisherPruner(BlockwiseWoodburryFisherPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(FlopsBlockwiseWoodburryFisherPruner, self).__init__(model, inp_args, **kwargs)
        logging.info("IN BLOCK WOODBURRY -> TARGET FLOPS")
        self._all_grads = None
        self._all_grads_dic = {}
        self._block_fisher_inv_dic = {}
        self._block_fisher_inv_diag_dic = {}
        self._param_stats = []
        # need to consider the update  - ( w_q  (H^-1 e_q ) / [H^-1]_qq
        assert not inp_args.prune_direction
        self._prune_direction = inp_args.prune_direction
        self._zero_after_prune = inp_args.zero_after_prune
        self._inspect_inv = inp_args.inspect_inv
        if self._inspect_inv:
            self.inspect_dic = {}
        self._fisher_mini_bsz = inp_args.fisher_mini_bsz
        if self._fisher_mini_bsz < 0:
            self._fisher_mini_bsz = 1
        self._total_block_fisher_params = 0
        self._flops_target = inp_args.flops_target * 1000000.0

        if hasattr(inp_args, 'fittable_params'):
            self._fittable_params = inp_args.fittable_params
        else:
            self._fittable_params = 64 * 64 * 3

        self._previous_epoch_level = None

    def _get_flops_stats(self):
        tmp_model = copy.deepcopy(self._model)
        # applies masks
        pruner_after_parameter_optimization_functional(tmp_model)
        # remove wrapings for use with flop counting hooks
        tmp_model = get_unwrapped_model(tmp_model)
        # minor hack for ResNet50, it seems somehow it got wrapped twice?
        tmp_model = get_unwrapped_model(tmp_model)

        total_flops, module_flops, module_names = get_macs_dpf(self.args, tmp_model, multiply_adds=False,
                                                               ignore_zero=True,
                                                               display_log=True,
                                                               ignore_bn=True, ignore_relu=True,
                                                               ignore_maxpool=True,
                                                               ignore_bias=True)
        del tmp_model

        return total_flops

    def prune_to_level(self, desired_flops_level, desired_sparsity_level, dset, subset_inds,
                       device, num_workers, epoch_num, before_sparsity, before_flops, **kwargs):
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

        if self.args.woodfisher_parallel!=-1 or self.args.woodfisher_parallel_all:
            self._parallel_compute_block_woodburry_fisher_inverse_all_layers()

        if self.args.flops_power > 0:
            assert self.args.woodburry_joint_sparsify
            flop_stat_dic = self._get_flop_stats()

        self._param_idx = 0

        #############################################################
        # Step1: compute blockwise wood fisher inverse!
        for idx, module in enumerate(self._modules):
            # print(f'module is {module}')
            self._compute_block_woodburry_fisher_inverse(module.weight, self._param_idx, device)
            self._all_grads_dic.pop(self._param_idx, None)
            # del self._all_grads_dic[self._param_idx]
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

            assert self.args.woodburry_joint_sparsify


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

        assert self._weight_only

        #############################################################
        # Step2: Doing global parameter selection!

        if desired_sparsity_level < 0:
            obtained_flops = before_flops

            lower_sparsity = before_sparsity
            upper_sparsity = 1
            max_iter_loop = 50 # to avoid getting stuck in an infinite loop
            loop_count = 0
            while abs(obtained_flops - desired_flops_level)/1000000.0 >= self.args.flops_epsilon and loop_count <= max_iter_loop:

                middle_sparsity = (lower_sparsity + upper_sparsity)/2

                logging.info(f"binary search: lower {lower_sparsity}, upper {upper_sparsity}, middle {middle_sparsity}")
                logging.info(f"binary search: obtained_flops {obtained_flops}, desired_flops {desired_flops_level}")

                global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), middle_sparsity)
                logging.info(f'shape of global param mask is {list(global_param_mask.shape)}')

                # apply global_param_masks and do weight update
                _param_count = 0
                for idx, module in enumerate(self._modules):
                    weight_mask_before = module.weight_mask
                    module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(
                        module.weight)

                    if self.args.fisher_cpu:
                        weight_mask_before = weight_mask_before.cpu()

                    pruned_weights = weight_mask_before - module.weight_mask
                    prune_mask = weight_mask_before > module.weight_mask  # active if was 1 before and 0 now
                    pruned_weights = pruned_weights.flatten().float()

                    # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
                    if not self.args.fisher_cpu:
                        scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                                _param_count)
                    else:
                        scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(),
                                                                                module.weight.flatten().cpu(),
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

                    logging.info(
                        f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                    with torch.no_grad():
                        module.weight[:] = module.weight.data + weight_update

                    logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight update is {torch.norm(weight_update).item()}')

                    logging.info(f'for param {idx} named {self._module_names[idx]} after update: norm of weight is {torch.norm(module.weight).item()}')

                    _param_count += module.weight.numel()

                    del pruned_weights
                    del prune_mask
                    del scaled_basis_vector
                    del weight_update
                    del weight_mask_before
                assert self._param_idx == _param_count

                # check the resulting flops
                obtained_flops = self._get_flops_stats()

                if desired_flops_level > obtained_flops:
                    # we pruned more extensively than needed
                    # so next time prune less
                    upper_sparsity = middle_sparsity
                elif desired_flops_level < obtained_flops:
                    # we pruned less extensively than needed
                    # so next time prune more
                    lower_sparsity = middle_sparsity

                logging.info(f"Reached FLOPs {obtained_flops} and target was {desired_flops_level}, at overall sparsity {middle_sparsity}")
                loop_count += 1
        else:

            global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), desired_sparsity_level)
            logging.info(f'shape of global param mask is {list(global_param_mask.shape)}')

            # apply global_param_masks and do weight update
            _param_count = 0
            for idx, module in enumerate(self._modules):
                weight_mask_before = module.weight_mask
                module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(
                    module.weight)

                if self.args.fisher_cpu:
                    weight_mask_before = weight_mask_before.cpu()

                pruned_weights = weight_mask_before - module.weight_mask
                prune_mask = weight_mask_before > module.weight_mask  # active if was 1 before and 0 now
                pruned_weights = pruned_weights.flatten().float()

                # print(f'weights in parameter {idx} named {self._module_names[idx]} before pruning (only for pruned) are ', module.weight[prune_mask])
                if not self.args.fisher_cpu:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights, module.weight.flatten(),
                                                                            _param_count)
                else:
                    scaled_basis_vector = self._get_pruned_wts_scaled_basis(pruned_weights.cpu(),
                                                                            module.weight.flatten().cpu(),
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

                logging.info(
                    f'for param {idx} named {self._module_names[idx]} before update: norm of weight is {torch.norm(module.weight).item()}')

                with torch.no_grad():
                    module.weight[:] = module.weight.data + weight_update

                logging.info(f'for param {idx} named {self._module_names[idx]}: norm of weight update is {torch.norm(weight_update).item()}')

                logging.info(
                    f'for param {idx} named {self._module_names[idx]} after update: norm of weight is {torch.norm(module.weight).item()}')

                _param_count += module.weight.numel()

                del pruned_weights
                del prune_mask
                del scaled_basis_vector
                del weight_update
                del weight_mask_before

            assert self._param_idx == _param_count

            obtained_flops = self._get_flops_stats()

        # if there is some issue in memory, make sure that at each param_count the below are None
        # self._block_fisher_inv_dic[_param_count] = None
        # self._block_fisher_inv_diag_dic[_param_count] = None

        # check if all the params whose fisher inverse was computed their value has been taken
        # print(f'param_idx is {self._param_idx} and fisher_inv_shape[0] is {len(self._all_grads[0])} \n')
        assert self._param_idx == self._total_block_fisher_params
        assert self._num_params == self._total_block_fisher_params

        del self._param_stats
        del self._block_fisher_inv_diag_dic
        del self._block_fisher_inv_dic

        if self._inspect_inv:
            meta['inspect_dic'] = self.inspect_dic
        return True, meta, obtained_flops

    def _get_sparsity_info(self, nick=""):
        if nick!="":
            nick += ": "
        sparsity_dict = {}
        total_num_zeros = 0
        total_num_params = 0
        for _name, _module in zip(self._module_names, self._modules):
            num_zeros, num_params = get_total_sparsity(_module)
            total_num_zeros += num_zeros
            total_num_params += num_params
            logging.info(f'layer named {_name} has {num_zeros} zeros out of total {num_params} params, % is {(num_zeros*1.0)/num_params}')
            sparsity_dict[_name] = (num_zeros, num_params)
        print(f"{nick}this is sparsity_dict", sparsity_dict)
        return total_num_zeros, total_num_params, sparsity_dict

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

        if epoch_num == self._start:
            # in the beginning start with pruning to initial_sparsity
            sparsity_level = self._required_sparsity(epoch_num)
            flops_level = -1
            self._before_flops = self._get_flops_stats()

        else:
            # else determine the FLOP count to remove
            sparsity_level = -1
            flops_level = self._required_flops(epoch_num)

        logging.info(f"Sparsity target at epoch {epoch_num} based on polynomial schedule is {sparsity_level}")
        logging.info(f"Flop target at epoch {epoch_num} based on polynomial schedule is {flops_level}")

        before_num_zeros, before_num_params, before_sparsity_dict = self._get_sparsity_info(nick="before pruning")
        before_sparsity = float(before_num_zeros)/before_num_params

        logging.info(f"At epoch {epoch_num}, sparsity is {before_sparsity} and FLOP count is {self._before_flops}")

        flag, global_meta, obtained_flops = self.prune_to_level(flops_level, sparsity_level, dset, subset_inds, device, num_workers, epoch_num, before_sparsity=before_sparsity, before_flops=self._before_flops, **kwargs)

        self._before_flops = obtained_flops

        if epoch_num == self._start:
            self._flops_init = obtained_flops

        return flag, global_meta