import logging
from utils.utils import flatten_tensor_list
from policies.pruners import GradualPruner

class MagnitudePruner(GradualPruner):
    def __init__(self, model, inp_args, **kwargs):
        super(MagnitudePruner, self).__init__(model, inp_args, **kwargs)
        self._prune_direction = inp_args.prune_direction

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        # magnitude pruner with stat as magnitude
        # here 1e-4 is just a small constant epsilon added to param stat to avoid all zeros
        return (param.abs() + 1e-4) * param_mask

    def on_epoch_begin(self, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}
        weight_mask_before = None

        for module in self._modules:
            weight_mask_before = module.weight_mask
            level = self._required_sparsity(epoch_num)
            # the w_stat and b_stat are then compared to a threshold, based on which they are pruned
            # it is mainly the param_stat that changes across various pruning methods
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)

        if self._prune_direction:
            meta['prune_direction'] = module.weight.data * (module.weight_mask - weight_mask_before).float()
            meta['original_param'] = module.weight.data.clone()
            meta['mask_previous'] = weight_mask_before
            meta['mask_overall'] = module.weight_mask
            print('previous mask is ', meta['mask_previous'])
            print('overall mask is ', meta['mask_overall'])

        return True, meta

class GlobalMagnitudePruner(MagnitudePruner):
    '''
    Prunes based on the magnitude of parameters like for the "MagnitudePruner",
    The difference is just that for global magnitude the statistic of all parameter tensors
    will be compared jointly/globally across all layers
    '''

    @staticmethod
    def _get_param_stat(param, param_mask, flop_stat=None):
        '''
        Takes as input the "param" tensor (think weight matrix) as well as its corresponding mask.
        The flop_stat option allows to use the FLOP-based pruner, which basically reweighs
        the pruning statistic based on how much FLOP reduction will occur when a parameter is set to zero
        from this weight matrix "param
        '''
        if param is None or param_mask is None: return None


        if flop_stat is not None:
            pruning_stat = (param.abs() + 1e-4) / (flop_stat + 1e-10)
        else:
            pruning_stat = (param.abs() + 1e-4)

        return pruning_stat * param_mask

    def on_epoch_begin(self, epoch_num, **kwargs):
        meta = {}
        if self._pruner_not_active(epoch_num):
            return False, {}

        if not hasattr(self, '_param_stats'):
            self._param_stats = []

        assert self._weight_only

        if self.args.flops_power > 0:
            flop_stat_dic = self._get_flop_stats()

        for idx, module in enumerate(self._modules):
            if self.args.flops_power > 0 and flop_stat_dic is not None:
                flop_stat = flop_stat_dic[self._module_names[idx]]
            else:
                flop_stat = None

            # save the w_stat across all layers
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask, flop_stat=flop_stat),\
                             self._get_param_stat(module.bias, module.bias_mask)

            self._param_stats.append(w_stat.flatten())

        level = self._required_sparsity(epoch_num)

        # decide which to prune globally
        global_param_mask = self._get_pruning_mask(flatten_tensor_list(self._param_stats), level)
        del self._param_stats

        # obtain the layerwise masks from this global mask
        _param_count = 0
        for idx, module in enumerate(self._modules):
            module.weight_mask = global_param_mask[_param_count:_param_count + module.weight.numel()].view_as(
                module.weight)
            _param_count += module.weight.numel()
            module.bias_mask = None

        del global_param_mask

        return True, meta
