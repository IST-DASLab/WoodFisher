"""
This .py contains various approximation schemes for pruning
differences. We assume that only weights are sparsified
"""
import torch
import torch.nn.functional as F

from utils import percentile
from copy import deepcopy


def magnitude(delta):
    raise NotImplementedError

def diag_hess(delta):
    raise NotImplementedError

def fisher(delta):
    raise NotImplementedError

def get_fisher(dset, model):
    raise NotImplementedError

def get_diag_hess(dset, model):
    raise NotImplementedError

def get_magnitude(dset, model):
    raise NotImplementedError


class Comparator:
    """
    Class that implements check for approximation 
    quality of the given method
    """
    def __init__(self, model, subset, device, modules, mode='magnitude'):
        """
        Arguments:
            model {nn.Module}: the pruned model
            subset {torch.utils.data.DataLoader}: training subset to measure statistics
            device {torch.device}: device to store the model/data
            modules {list of str}: layer instances to prune
            mode {str}: the approximation method to use

        Notes:
            mode can take values in {'magnitude', 'diag_hess', 'fisher'}
        """
        self._model = model
        self._subset = subset
        self._mode = mode
        self._module_names = modules
        self._device = device

        self._get_hess_approx = lambda dset, model: globals()[f'get_{self._mode}'](dset, model)
        self._get_approximation = lambda delta: globals()[self._mode](delta)

    def _get_masked_model(self, mask, module_name):
        masked_model = deepcopy(self._model)
        layer = dict(self._model.named_modules())[module_name]
        layer.weight.data *= mask
        return masked_model

    def _get_true_diff(self):
        pass

    def _get_nll(self, model):
        nll_loss = 0.
        with torch.no_grad():
            for in_tensor, target in self._subset:
                in_tensor, target = in_tensor.to(self._device), target.to(self._device)
                output = model(in_tensor)
                nll_loss += F.cross_entropy(output, target, reduction='sum').item()
        nll_loss /= len(self._subset.dataset)
        return nll_loss

    def _get_mask(self):
        raise NotImplementedError

    def _get_true_diff(self, masked_model):
        raise NotImplementedError

    def _get_approx_diff(self, masked_model):
        raise NotImplementedError

    def run_comparison(self):
        for module_name in self._module_names:
            mask = self._get_mask()
            masked_model = self._get_masked_model(mask, module_name)

            true_diff = self._get_true_diff()
            approx_diff = self._get_approx_diff()


