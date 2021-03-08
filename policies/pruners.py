"""
Implement Pruners here.

"""
import numpy as np

from policies.policy import PolicyBase
from utils import (get_total_sparsity, 
                    recompute_bn_stats,
                    percentile,
                    get_prunable_children,
                    flatten_tensor_list,
                    dump_tensor_to_mat,
                    add_outer_products_efficient_v1,
                    get_summary_stats)

import sys
PATH_TO_HESSIAN='/nfs/scistore14/alistgrp/ssingh/projects/hessian/'
PATH_TO_HESSIANFLOW = '/nfs/scistore14/alistgrp/ssingh/projects/ihvp/'
sys.path.append(PATH_TO_HESSIAN)
sys.path.append(PATH_TO_HESSIANFLOW)

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import logging
from typing import List, Dict
from copy import deepcopy
import time
from tqdm import tqdm
import math
import os
import json
import copy
from utils.checkpoints import get_unwrapped_model
from utils.pruner_utils import pruner_after_parameter_optimization_functional
from utils.flop_utils import get_macs_dpf

def build_pruner_from_config(model, pruner_config, inp_args=None):
    """
    This function takes the takes pruner config and model that are provided by function
    build_pruners_from_config. We assume that each pruner have one parameter
    group, i.e., which shares sparsity levels and pruning schedules.

    The *suggested!* .yaml file structure is defined in build_pruners_from_config.
    """
    pruner_class = pruner_config['class']
    pruner_args = {k: v for k, v in pruner_config.items() if k != 'class'}
    # globals returns a dictionary containing the variables defined in the global namespace
    pruner = globals()[pruner_class](model, inp_args, **pruner_args)
    return pruner

def build_pruners_from_config(model, config, inp_args=None):
    """
    This function takes *general* config file for current run and model 
    and returns a list of pruners which are build by build_pruner_from_config.

    Example config.yaml for pruner instances:

    >>> pruners:
    >>>   pruner_1:
    >>>     class: MagnitudePruner
    >>>     epochs: [0,2,4] # [start, freq, end] for now
    >>>     weight_only: True # if True prunes only *.weight parameters in specified layers
    >>>                       # if *.bias is None thif flag is just ignored
    >>>     initial_sparsity: 0.05 # initial sparsity level for parameters
    >>>     target_sparsity: 0.7 # desired sparsity level at the end of pruning
    >>>     modules: [net.0] # modules of type (nn.Conv2d, nn.Linear)
    >>>     keep_pruned: False # Optional from now on
    >>>     degree: 3 # Optional degree to use for polynomial schedule
    >>>   pruner_2:
    >>>     class: MagnitudePruner
    >>>     epochs: [0,2,4]
    >>>     weight_only: True
    >>>     initial_sparsity: 0.05
    >>>     target_sparsity: 0.8
    >>>     modules: [net.2]
    >>>     keep_pruned: False


    There is an optional arguments:
        keep_pruned: whether pruned weights values shoud be store, recommended values is false 
                     unless you want to use reintroduction with previous magnitudes
    """
    if 'pruners' not in config: return []
    pruners_config = config['pruners']
    pruners = [build_pruner_from_config(model, pruner_config, inp_args=inp_args)
               for pruner_config in pruners_config.values()]
    return pruners


class Pruner(PolicyBase):
    def __init__(self, *args, **kwargs):
        assert hasattr(self, '_modules'), "@Pruner: make sure any Pruner has 'modules' and 'module_names' attribute"
        assert hasattr(self, '_module_names'), "@Pruner: make sure any Pruner has 'modules' and 'module_names' attribute"
        # this is needed because after_parameter_optimization method assumes this:
        assert all([is_wrapped_layer(_module) for _module in self._modules]), \
            "@Pruner: currently the code assumes that you supply prunable layers' names directly in the config"

    def on_epoch_end(self, **kwargs):
        sparsity_dict = {}
        for _name, _module in zip(self._module_names, self._modules):
            num_zeros, num_params = get_total_sparsity(_module)
            print(f'layer named {_name} has {num_zeros} zeros out of total {num_params} params')
            sparsity_dict[_name] = (num_zeros, num_params)
        return sparsity_dict

    def after_parameter_optimization(self, model, **kwargs):
        """
        Currently this stage is used to mask pruned neurons within the layer's data.
        it is GradualPruners' method only.
        """
        for _module in self._modules:
            if 'ReLU' not in str(type(_module)): 
                _module.apply_masks_to_data()

class GradualPruner(Pruner):
    def __init__(self, model, inp_args, **kwargs):
        """
        Arguments:
            model {nn.Module}: network with wrapped modules to bound pruner
        Key arguments:
            kwargs['initial_sparsity']: initial_sparsity layer sparsity
            kwargs['target_sparsity']: target sparsity for pruning end
            kwargs['weight_only']: bool, if only weights are pruned
            kwargs['epochs']: list, [start_epoch, pruning_freq, end_epoch]
            kwargs['modules']: list of module names to be pruned
            kwargs['degree']: float/int, degree to use in polinomial schedule, 
                              degree == 1 stands for uniform schedule
        """
        self._start, self._freq, self._end = kwargs['epochs']
        self._weight_only = kwargs['weight_only']
        self._initial_sparsity = kwargs['initial_sparsity']
        self._target_sparsity = kwargs['target_sparsity']
        self.args = inp_args
        self._keep_pruned = kwargs['keep_pruned'] if 'keep_pruned' in kwargs else False
        self._degree = kwargs['degree'] if 'degree' in kwargs else 3

        self._model = model
        modules_dict = dict(self._model.named_modules())

        prefix = ''
        # This is the thing which is causing the issue!!
        if not self.args.ignore_prefix:
            if isinstance(self._model, torch.nn.DataParallel):
                prefix = 'module.'
        # Unwrap user-specified modules to prune into lowest-level prunables:
        # _module_names contains the names of the layers that we want to prune!
        self._module_names = [prefix + _name for _name in kwargs['modules']]
        # self._module_names = [prefix + _name for _name in get_prunable_children(self._model, kwargs['modules'])]
        # self._bn_module_names = {}

        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]

        if self._keep_pruned:
            for module in self._modules:
                module.copy_pruned(True)

        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k,v in kwargs.items()]) + '\n')

    def update_initial_sparsity(self):
        parameter_sparsities = []
        for module in self._modules:
            w_sparsity, b_sparsity = module.weight_sparsity, module.bias_sparsity
            parameter_sparsities.append(w_sparsity)
            if b_sparsity is not None: parameter_sparsities.append(b_sparsity)
        self._initial_sparsity = np.mean(parameter_sparsities)

    @staticmethod
    def _get_param_stat(param):
        raise NotImplementedError("Implement in child class.")

    def _get_flop_stats(self):
        tmp_model = copy.deepcopy(self._model)
        # applies masks
        pruner_after_parameter_optimization_functional(tmp_model)
        # remove wrapings for use with flop counting hooks
        tmp_model = get_unwrapped_model(tmp_model)
        # minor hack for ResNet50 (it seems somehow it got wrapped twice)
        tmp_model = get_unwrapped_model(tmp_model)

        total_flops, module_flops, module_names = get_macs_dpf(self.args, tmp_model, multiply_adds=False, ignore_zero=True,
                                                               display_log=True,
                                                               ignore_bn=True, ignore_relu=True, ignore_maxpool=True,
                                                               ignore_bias=True)
        del tmp_model

        # remove the beginning '.'
        module_names = [name.replace('.module', 'module') if '.module' in name else name for name in module_names]

        module_param_count_dic = {}
        # divide the layerwise FLOP cost by the number of parameters in that layer
        if self.args.flops_per_param:
            for idx, module in enumerate(self._modules):
                print(f"In module idx {idx} named {self._module_names[idx]}, # params is {module.weight.numel()} "
                      f"and # non-zero are {(module.weight.data != 0).float().sum()}")
                module_param_count_dic[self._module_names[idx]] = (module.weight.data != 0).float().sum()

            for idx, name in enumerate(module_names):
                # the module_names might be different from the self._modules
                # since not all layers might be under consideration for pruning
                if name in module_param_count_dic:
                    module_flops[idx] /= module_param_count_dic[name]

        # ------------ The flops_normalize can be IGNORED ------------
        # do additional normalizations to put the flops in the right scale
        if self.args.flops_normalize == 'million':
            module_flops = [1.0 * flop / 1e6 for flop in module_flops]
        elif self.args.flops_normalize == 'log':
            module_flops = [math.log2(flop) for flop in module_flops]
        # ------------------------------------------------------------

        # flops_power is a hyperparameter that will control the important of this flop based pruning statistic
        module_flops = [flop ** self.args.flops_power for flop in module_flops]

        flop_dict = dict(zip(module_names, module_flops))
        logging.info(f"Flop_dict is {flop_dict}")

        return flop_dict

    def _polynomial_schedule(self, curr_epoch):
        scale = self._target_sparsity - self._initial_sparsity
        progress = min(float(curr_epoch - self._start) / (self._end - self._start), 1.0)
        remaining_progress = (1.0 - progress) ** self._degree
        return self._target_sparsity - scale * remaining_progress

    def _polynomial_schedule_flops(self, curr_epoch):
        scale = self._flops_target - self._flops_init
        progress = min(float(curr_epoch - self._start) / (self._end - self._start), 1.0)
        remaining_progress = (1.0 - progress) ** self._degree
        return self._flops_target - scale * remaining_progress

    def _required_sparsity(self, curr_epoch):
        return self._polynomial_schedule(curr_epoch)

    def _required_flops(self, curr_epoch):
        return self._polynomial_schedule_flops(curr_epoch)

    def _pruner_not_active(self, epoch_num):
        # PR: This doesn't support one-shot pruning after 1 epoch of training
        # For one-shot after n epochs one can set epochs = [0, n-1, n]
        # but won't work when n = 1, as due to 0%0 in the line below!
        print(" in _pruner_not_activate: epoch_num is ", epoch_num)
        if epoch_num == 0:
            if self.args.prune_at_launch:
                return False
            elif self._initial_sparsity <=0:
                return True
        if epoch_num == self._start and self._initial_sparsity <=0:
            return True
        return ((epoch_num - self._start) % self._freq != 0 or epoch_num > self._end or epoch_num < self._start)


    @staticmethod
    def _get_pruning_mask(param_stats, sparsity):
        if param_stats is None: return None
        threshold = percentile(param_stats, sparsity)
        return (param_stats > threshold).float()
        

# to have the pruner names in the globals()
from pruners.woodfisher import WoodburryFisherPruner
from pruners.woodfisherblock import BlockwiseWoodburryFisherPruner
from pruners.woodtaylor import WoodburryTaylorPruner
from pruners.woodtaylorblock import BlockwiseWoodburryTaylorPruner
from pruners.woodfisherblockgroup import GroupBlockwiseWoodburryFisherPruner
from pruners.woodfisherblockdynamic import DynamicBlockwiseWoodburryFisherPruner
from pruners.kfac import KFACFisherPruner
from pruners.magnitude_based import MagnitudePruner, GlobalMagnitudePruner
from pruners.diagfisher import FisherPruner
from pruners.naivehess import NaiveHessianPruner
from pruners.woodfisherblock_flops import FlopsBlockwiseWoodburryFisherPruner

if __name__ == '__main__':
    pass
