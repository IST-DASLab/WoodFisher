"""
Implementations for reintroducing policies for weight/kernels 
"""
import numpy as np

from policies.policy import PolicyBase
from utils import (get_total_sparsity, 
                   recompute_bn_stats, 
                   percentile, 
                   is_wrapped_layer,
                   get_normal_stats)

import torch
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

import logging
from typing import List, Dict

def build_recycler_from_config(model, recycler_config):
    """
    This function takes the takes pruner config and model that are provided by function
    build_recyclers_from_config. We assume that each recycler have one parameter
    group, i.e., which shares sparsity levels and recycling schedules.

    The *suggested!* .yaml file structure is defined in build_pruners_from_config.
    """
    recycler_class = recycler_config['class']
    recycler_kwargs = {k: v for k, v in recycler_config.items() if k != 'class'}
    recycler = globals()[recycler_class](model, **recycler_kwargs)
    return recycler

def build_recyclers_from_config(model, config):
    """
    This function takes *general* config file for current run and model 
    and returns a list of recyclers which are build by build_recycler_from_config.

    Example config.yaml for pruner instances:

    >>> recyclers:
    >>>   recycler_1:
    >>>     class: RandomRecycler
    >>>     epochs: [0,2,4] # [start, freq, end] for now
    >>>     weight_only: True # if True recycles only *.weight parameters in specified layers
    >>>                       # if *.bias is None thif flag is just ignored
    >>>     initial_density: 0.05 # initial density level for parameters
    >>>     target_density: 0.7 # desired density level at the end of pruning
    >>>     modules: [net.0] # modules of type (nn.Conv2d, nn.Linear)
    >>>     degree: 3 # Optional degree to use for polynomial schedule
    >>>   recycler_2:
    >>>     class: RandomRecycler
    >>>     epochs: [0,2,4]
    >>>     weight_only: True
    >>>     initial_density: 0.05
    >>>     target_density: 0.8
    >>>     modules: [net.2]


    There is an optional arguments:
        keep_pruned: whether pruned weights values shoud be store, recommended values is false 
                     unless you want to use reintroduction with previous magnitudes
    """
    if 'recyclers' not in config: return []
    recyclers_config = config['recyclers']
    recyclers = [build_recycler_from_config(model, recycler_config)
                 for recycler_config in recyclers_config.values()]
    return recyclers


class GradualRecycler(PolicyBase):
    def __init__(self, model, **kwargs):
        """
        Arguments:
            model {nn.Module}: network with wrapped modules to bound pruner
        Key arguments:
            kwargs['initial_density']: initial layer density
            kwargs['target_sparsity']: target layer density
            kwargs['weight_only']: bool, if only weights are reintroduced 
                                   is placed according to corresponding layer pruner
            kwargs['epochs']: list, [start_epoch, pruning_freq, end_epoch]
            kwargs['modules']: list of module names to be recycled
            kwargs['degree']: float/int, degree to use in polinomial schedule, 
                              degree == 1 stands for uniform schedule
        """
        self._start, self._freq, self._end = kwargs['epochs']
        self._weight_only = kwargs['weight_only']
        self._initial_density = kwargs['initial_density']
        self._target_density = kwargs['target_density']
        self._degree = kwargs['degree'] if 'degree' in kwargs else 3

        self._model = model
        modules_dict = dict(self._model.named_modules())

        prefix = ''
        if isinstance(self._model, torch.nn.DataParallel):
            prefix = 'module.'
        self._module_names = [prefix + _name for _name in kwargs['modules']]
        
        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]

        assert hasattr(self, '_modules'), "@Recycler: make sure any Recycler has 'modules' and 'module_names' attribute"
        assert hasattr(self, '_module_names'), "@Recycler: make sure any Recycler has 'modules' and 'module_names' attribute"
        assert all([is_wrapped_layer(_module) for _module in self._modules]), \
            "@Recycler: currently the code assumes that you supply prunable layers' names directly in the config"

        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k,v in kwargs.items()]) + '\n')

    def after_parameter_optimization(self, **kwargs):
        for _module in self._modules:
            _module.apply_masks_to_data()

    @staticmethod
    def _get_param_stat(param):
        raise NotImplementedError("Implement in child class.")

    def _polynomial_schedule(self, curr_epoch):
        scale = self._target_density - self._initial_density
        progress = min(float(curr_epoch - self._start) / (self._end - self._start), 1.0)
        remaining_progress = (1.0 - progress) ** self._degree
        return self._target_density - scale * remaining_progress

    def _required_density(self, curr_epoch):
        return 1 - self._polynomial_schedule(curr_epoch)

    def _recycler_not_active(self, epoch_num):
        return ((epoch_num - self._start) % self._freq != 0 or epoch_num > self._end or epoch_num < self._start)

    @staticmethod
    def _get_pruning_mask(param_stats, density):
        if param_stats is None: return None
        threshold = percentile(param_stats, density)
        return (param_stats > threshold).float()

    @staticmethod
    def _init_recycled(module):
        raise NotImplemented('For now _init_recycled is defined for each child recycler separately')


class RandomRecycler(GradualRecycler):
    def __init__(self, model, **kwargs):
        super(RandomRecycler, self).__init__(model, **kwargs)
        self.opt_list = None

    @staticmethod
    def _get_param_stat(param, param_mask):
        if param is None or param_mask is None: return None
        param_stat = torch.zeros_like(param).uniform_()
        param_stat[param_mask.byte()] = 1e4
        return param_stat

    #WARNING: supports only one parameter group currently
    def _preproc_dict(self, optimizer):
        self.opt_list = [[None,None]]*len(self._modules)
        module_idx, flag_bias = 0, False
        for param in optimizer.state.keys():
            if module_idx == len(self._modules): break
            module = self._modules[module_idx]
            if flag_bias:
                if module.bias.size() != param.size(): continue
            else:
                if module.weight.size() != param.size(): continue
            if flag_bias:
                if (module.bias == param).sum() == param.numel():
                    self.opt_list[module_idx][1] = param
                    module_idx += 1
                    flag_bias = False
                    continue
            if (module.weight == param).sum() == param.numel():
                self.opt_list[module_idx][0] = param
                if module.bias is not None:
                    flag_bias = True
                else:
                    module_idx += 1

    @staticmethod
    def _init_recycled_module(module):
        if module.bias_mask is not None and module.bias is not None:
            byte_mask = module.bias_mask.byte()
            mu, sigma = get_normal_stats(module.bias[byte_mask])
            module.bias.data[~byte_mask] =  module.bias.data[~byte_mask].normal_().mul_(sigma).add_(mu)
        byte_mask = module.weight_mask.byte()
        mu, sigma = get_normal_stats(module.weight[byte_mask])
        module.weight.data[~byte_mask] = module.weight.data[~byte_mask].normal_().mul_(sigma).add_(mu)

    def _init_recycled(self):
        for module in self._modules:
            self._init_recycled_module(module)
  
    def _apply_momentum_mask(self, optimizer, module_idx, weight_mask, bias_mask):
        optimizer.state[self.opt_list[module_idx][0]]['momentum_buffer'] *= weight_mask
        if bias_mask is not None:
            optimizer.state[self.opt_list[module_idx][1]]['momentum_buffer'] *= bias_mask
    
    def on_epoch_begin(self, epoch_num, optimizer, **kwargs):
        meta = {}
        if self._recycler_not_active(epoch_num):
            return False, {}
        if self.opt_list is None:
            self._preproc_dict(optimizer)
        self._init_recycled()
        for module_idx, module in enumerate(self._modules):
            level = self._required_density(epoch_num)
            self._apply_momentum_mask(optimizer, module_idx, module.weight_mask, module.bias_mask)
            w_stat, b_stat = self._get_param_stat(module.weight, module.weight_mask),\
                             self._get_param_stat(module.bias, module.bias_mask)
            module.weight_mask, module.bias_mask = self._get_pruning_mask(w_stat, level),\
                                                   self._get_pruning_mask(None if self._weight_only else b_stat, level)
            
        return True, meta


if __name__ == '__main__':
    pass