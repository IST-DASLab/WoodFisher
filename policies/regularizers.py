"""
Implement regularization policies here
"""
import torch
import torch.nn as nn

from policies.policy import PolicyBase

import logging

def build_reg_from_config(model, reg_config):
    """
    This function build regularizer given the model (only need for weigths typically)
    and regularizer configuration.
    """
    reg_class = reg_config['class']
    reg_args = {k: v for k, v in reg_config.items() if k != 'class'}
    reg = globals()[reg_class](model, **reg_args)
    return reg

def build_regs_from_config(model, config):
    """
    This function takes *general* config file for current run and model 
    and returns a list of regularizers which are build by build_reg_from_config.

    Example config.yaml for pruner instances:

    >>> regularizers:
    >>>   reg1:
    >>>     class: Hoyer # regularization method to use
    >>>     lambda: 1e-6 # regularization coefficient 
    >>>     modules: [net.0] # modules to apply regularization
    >>>     weight_only: True # if regularizer is applied only to weights of module (no bias)
    >>>   reg2:
    >>>     class: HoyerSquare
    >>>     lambda: 1e-6
    >>>     modules: [net.2]
    >>>     weight_only: True
    """
    if 'regularizers' not in config: return []
    regs_config = config['regularizers']
    regs = [build_reg_from_config(model, reg_config)
            for reg_config in regs_config.values()]
    return regs


class Regularizer(PolicyBase):
    def __init__(self, model, **kwargs):
        self._model = model
        if not isinstance(self._model, nn.Module):
            raise ValueError('model should be an instance of nn.Module')
        modules_dict = dict(self._model.named_modules())
        self._weight_only, self._lambda = kwargs['weight_only'], float(kwargs['lambda'])

        prefix = ''
        if isinstance(self._model, torch.nn.DataParallel):
            prefix = 'module.'
        self._module_names = [prefix + _name for _name in kwargs['modules']]

        self._modules = [
            modules_dict[module_name] for module_name in self._module_names
        ]

        logging.debug(f'Constructed {self.__class__.__name__} with config:')
        logging.debug('\n'.join([f'    -{k}:{v}' for k,v in kwargs.items()]) + '\n')

    def on_minibatch_begin(self, **kwargs):
        reg_loss = self._lambda * self._compute_penalty()
        return reg_loss

    def _compute_penalty(self):
        """
        Base regularizer method which computes regularization penalty.
        """
        raise ValueError('Implement in a child class')


class Hoyer(Regularizer):
    def __init__(self, model, **kwargs):
        super(Hoyer, self).__init__(model, **kwargs)

    def _compute_penalty(self):
        penalty = 0.
        for module in self._modules:
            dim = module.weight.numel()
            vector = module.weight.view(-1)
            if not self._weight_only and module.bias:
                dim += self.module.bias.numel()
                vector = torch.cat([vector, module.bias])
            penalty += (dim ** 0.5 - vector.abs().sum() / vector.norm()) / (dim ** 0.5 - 1)
        return penalty


class SquaredHoyer(Regularizer):
    def __init__(self, model, **kwargs):
        super(SquaredHoyer, self).__init__(model, **kwargs)

    def _compute_penalty(self):
        penalty = 0.
        for module in self._modules:
            vector = module.weight.view(-1)
            if not self._weight_only and module.bias:
                vector = torch.cat([vector, module.bias])
            penalty += vector.abs().sum().pow(2) / vector.norm().pow(2)
        return penalty


if __name__ == '__main__':
    pass