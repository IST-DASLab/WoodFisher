"""
Utility functions and classes to work with pruning.

In the end, outside modules should primarily:
* use get_wrapped_model,
* potentially add functionality to Wrapped layer,
and not worry about the rest.

"""

import torch
import torch.nn as nn
from copy import deepcopy
import logging

# Lowest level prunable modules:
PRUNABLE_MODULES = (
    nn.Conv2d, nn.Linear
)


def get_prunable_children(model, name_list):
    """
    Function that accepts super-modules that the user specified for pruner to prune,
    and unwraps these names into lowest-level prunable entities (possibly just themselves).

    This function makes inputting prunable layers into config files more user-friendly.
    """
    res = []
    for upper_name, upper_module in model.named_modules():
        if upper_name not in name_list:
            continue
        # iterate over children, itself included:
        # this can already be wrapped, then just use it:
        if is_wrapped_layer(upper_module):
            res.append(upper_name)
        else:
            # otherwise iterate over children
            for sub_name, sub_module in upper_module.named_modules():
                if is_wrapped_layer(sub_module):
                    res.append(f"{upper_name}.{sub_name}")
    return res


def num_zero(tensor):
    if tensor is None: return 0
    return (tensor == 0).int().sum()


def num_non_zero(tensor):
    if tensor is None: return 0
    return tensor.numel() - num_zero(tensor)


def num_total(tensor):
    if tensor is None: return 0
    return tensor.numel()


def is_wrapped_layer(layer: 'nn.Module') -> bool:
    """
    Use this function to test whether an instance
    of nn.Module is a prunable wrapped layer.
    """
    return isinstance(layer, WrappedLayer)


def should_prune_layer(layer: 'nn.Module') -> bool:
    """ Helper function """
    return isinstance(layer, PRUNABLE_MODULES)


def get_wrapped_model(model: 'nn.Module') -> 'nn.Module':
    """
    This function is the primary interface for outside code.
    Use as
    ```
    model = MyCustomPytorchModel(...pretrained=True...)
    model = get_wrapped_model(model)
    """
    assert not next(model.parameters()).is_cuda, "Model shouldn't be on CUDA, \
        as some parameters will be copied and you risk OOM."
    wrap_module(model)
    # for the Pruners to verify this function has been applied:
    model._was_wrapped = True
    return model


def wrap_module(module, prefix='.'):
    """
    Recursive function which iterates over PRUNABLE_MODULES of this
    module and wraps them with WrappedLayer class.
    """
    module_dict = dict(module.named_children())
    for name, sub_module in module_dict.items():
        # checks if it is there in the set named PRUNABLE_MODULES
        if should_prune_layer(sub_module):
            # model.name = Wrapped(sub_module)
            # vs previously model.name = sub_module
            setattr(module, name, WrappedLayer(sub_module))
            logging.debug(f'Module {prefix + name} was successfully wrapped')
            continue
        wrap_module(sub_module, prefix + name + '.')


class WrappedLayer(nn.Module):
    """
        Layer modules "wrapped" with containers/buffers to hold the masks
        and other info
    """
    def __init__(self, layer, *args, **kwargs):
        super(WrappedLayer, self).__init__()
        self._validate(layer) # sanity check
        self._layer = layer

        bias_mask_buffer = None 
        if self._layer.bias is not None:
            bias_mask_buffer = torch.ones_like(self._layer.bias).requires_grad_(False)
        self.register_buffer('_bias_mask', bias_mask_buffer)
        self.register_buffer('_weight_mask', torch.ones_like(self._layer.weight).requires_grad_(False))

        self.mask_grad_hooks = None

        self.last_activations = None
        self.trace_activations = False

        self._copy_pruned = False
        self._bias_data_copy = None
        self._weight_data_copy = None

    @property
    def weight(self):
        return self._layer.weight

    @weight.setter
    def weight(self, new_weight):
        self._layer.weight = new_weight

    @property
    def bias(self):
        # Throws an exception iff the .bias call downstream does
        return self._layer.bias

    @bias.setter
    def bias(self, new_bias):
        self._layer.bias = new_bias

    @staticmethod
    def _build_pruned_copy(param, new_mask, prev_copy=None):
        """
        Builds new copy of pruned weights after each time the
        *modify_masks* method is called (typically by a pruner
        on epoch_begin)

        WARNING!!! Assumption:  pruning masks updates are nested
        """
        if param is None or new_mask is None: return None
        if prev_copy is None: return param.data * (1 - new_mask)
        return prev_copy + param.data * (1 - new_mask)

    def _mask_setter(self, new_mask, name='weight'):
        param = getattr(self, name)
        param_data_copy = getattr(self, f'_{name}_data_copy')
        if self._copy_pruned:
            setattr(self, f'_{name}_data_copy', self._build_pruned_copy(param, new_mask, param_data_copy))
        setattr(self, f'_{name}_mask', new_mask)

    @property
    def weight_mask(self):
        return self._weight_mask

    @weight_mask.setter
    def weight_mask(self, new_mask):
        self._mask_setter(new_mask, 'weight')

    @property
    def bias_mask(self):
        return self._bias_mask

    @bias_mask.setter
    def bias_mask(self, new_mask):
        self._mask_setter(new_mask, 'bias')

    @staticmethod
    def _get_mask_grad(param):
        if param.grad is None:
            raise ValueError('Compute gradients w.r.t. parameter')

        return param.grad.mul(param)

    @property
    def weight_mask_grad(self):
        return self._get_mask_grad(self.weight)

    @property
    def bias_mask_grad(self):
        if self.bias_mask is None: return None
        return self._get_mask_grad(self.bias)

    @property
    def layer_sparsity(self):
        """
        Property which contains current sparsity level for a wrapped layer
        """
        return num_zero(self.weight_mask), num_zero(self.bias_mask),\
               num_total(self.weight_mask) + num_total(self.bias_mask)

    @property
    def weight_sparsity(self):
        return num_zero(self.weight_mask).float() / self.weight_mask.numel().float()

    @property
    def bias_sparsity(self):
        if self.bias_mask is None: return None
        return num_zero(self.bias_mask).float() / self.bias_mask.numel().float()

    def unwrap(self) -> 'nn.Module':
        return self._layer

    def apply_masks_to_data(self):
        if self.bias_mask is not None:
            self.bias.data *= self.bias_mask
        self.weight.data *= self.weight_mask

    def unmasked_forward(self, *args, **kwargs):
        return self._layer.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        self.apply_masks_to_data()

        activations = self._layer.forward(*args, **kwargs)
        if self.trace_activations: 
            self.last_activations = activations

        return activations

    def revert_pruned(self):
        """
        Revert's the pruned weights to their pre-pruning values.
        USE_CASE: when reintroducing the neurons.
        """
        if not self._copy_pruned:
            raise ValueError('No weights maintained to revert')
        if self.bias_mask is not None:
            revert_bias_mask = ~self.bias_mask.byte()
            self.bias.data[revert_bias_mask] = self.bias_data_copy[revert_bias_mask]
        revert_weight_mask = ~self.weight_mask.byte()
        self.weight.data[revert_weight_mask] = self.weight_data_copy[revert_weight_mask]

    def mask_grad(self):
        """
        Registers the hooks for masking the params gradients on backward.
        USE_CASE: can be invoked by user? pruner?
        """
        if self.mask_grad_hooks is not None:
            raise ValueError('Remove old hooks first')
        if self.bias_mask is not None:
            h_b = self.bias.register_hook(self.build_mask_grad_hook(self.mask_bias))
            self.mask_grad_hooks.append(h_b)
        h_w = self.weight.register_hook(self.build_mask_grad_hook(self.mask_weight))
        self.mask_grad_hooks.append(h_w)

    def unmask_grad(self):
        """
        Doing the direct opposite of previous if possible (i.e., hooks are present)
        """
        if self.mask_grad_hooks is None:
            raise ValueError('There is no parameter gradients masked')
        for h in self.mask_grad_hooks: 
            h.remove()
        self.mask_grad_hooks = None

    def trace_activations(self, enable: bool):
        """
        Calling the method enables/disables (depends on enable)
        the activation tracking
        """
        self.trace_activations = enable
        if not enable:
            self.last_activations = None

    def copy_pruned(self, enable: bool):
        """
        Calling the method enables/disables (depends on enable)
        maintaining the pruned values and deletes copies if called
        with enable=False (i.e., do not maintain)
        """
        self._copy_pruned = enable
        if not enable:
            self.weight_data_copy = None
            self.bias_data_copy = None

    @staticmethod
    def build_mask_grad_hook(mask):
        """
        Builds gradient masking hook
        """
        hook = lambda grad: grad.mul(mask)
        return hook

    @staticmethod
    def _validate(layer):
        assert should_prune_layer(layer), \
            "Can only prune modules that have base class nn.Conv2d or nn.Linear"


if __name__ == '__main__':
    pass

        


