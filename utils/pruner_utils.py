from utils.utils import get_total_sparsity
from utils.manager_utils import get_linear_conv_modules

def pruner_epoch_end_functional(model):

    _modules, _module_names = get_linear_conv_modules(model)
    sparsity_dict = {}
    for _name, _module in zip(_module_names, _modules):
        num_zeros, num_params = get_total_sparsity(_module)
        print(f'layer named {_name} has {num_zeros} zeros out of total {num_params} params')
        sparsity_dict[_name] = (num_zeros, num_params)
    return sparsity_dict

def pruner_after_parameter_optimization_functional(model):
    _modules, _ = get_linear_conv_modules(model)

    for _module in _modules:
        _module.apply_masks_to_data()