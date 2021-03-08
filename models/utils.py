"""
Utilities for models
"""

import torch
import torch.nn as nn
from models.blocks import get_2d_conv, round_to_power
from models.efficientnet import construct_effnet_for_dataset


def get_num_params(module):
    """ Compute number of parameters """
    s = 0
    for parameter in module.parameters():
        s += parameter.numel()
    return s


def get_module_by_name(model, name):
    """
    Helper method to get a nested child module from model,
    given a name of the form [child1].[child2].[...].module_name
    """
    if name == "":
        return model
    names = name.split('.')
    mod = model
    for n in names:
        mod = getattr(mod, n)
    return mod


def set_module_by_name(model, name, module):
    """
    Helper method to set a nested child module from model,
    given a name of the form [child1].[child2].[...].module_name
    """
    if name == '':
        model = module  # this is probably not correct
        return
    names = name.split('.')
    mod = model
    # Find the last one, whose attribute we want to set:
    for n in names[:-1]:
        mod = getattr(mod, n)
    setattr(mod, names[-1], module)


def replace_conv_with_butterfly(model, name_pair, device):
    name, nxt_name = name_pair
    conv = get_module_by_name(model, name)
    params = {
        'in_channels': round_to_power(conv.in_channels),
        'out_channels': round_to_power(conv.out_channels),
        'kernel_size': conv.kernel_size,
        'stride': conv.stride,
        'padding': conv.padding,
        'bias': conv.bias,
        'use_butterfly': conv.groups == 1,  # depthwise conv should remain Conv2d
        'groups': round_to_power(conv.groups)
    }
    butterfly_conv = get_2d_conv(**params)
    butterfly_conv.to(device)
    # Replace the attribute with newly created ButterflyConv:
    set_module_by_name(model, name, butterfly_conv)

    # If needed, correct the next module too:
    if nxt_name is not None and "_bn" in nxt_name:
        orig_bn = get_module_by_name(model, nxt_name)
        bn = nn.BatchNorm2d(
            num_features=round_to_power(conv.out_channels), 
            momentum=orig_bn.momentum,
            eps=orig_bn.eps)
        set_module_by_name(model, name, bn)


def get_next_module_name(model, name):
    if '.' not in name:
        parent_module = model
        children = list(parent_module.named_children())
        for i in range(len(children)):
            child_name, child_module = children[i]
            if i+1 < len(children) and child_name == name:
                return children[i+1][0]
        return None

    parent = ".".join(name.split('.')[:-1])
    parent_module = get_module_by_name(model, parent)
    children = list(parent_module.named_children())
    for i in range(len(children)):
        child_name, child_module = children[i]
        if i+1 < len(children) and parent+"."+child_name == name:
            return parent + "." + children[i+1][0]
    return None


def get_ordered_list_of_convs(model):
    """Extracts layers to be replaced by Butterfly convolution"""
    res = []
    named_modules = list(model.named_modules())
    for i, (name, module) in enumerate(named_modules):
        # this works on Conv2dStaticSamePadding and Conv2dDynamicSamePadding, too:
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nxt_name = get_next_module_name(model, name)
            res.append((name, nxt_name))
    # First convolutional layer shouldn't be replaced:
    res.pop(0)
    return res


def get_effnet_clone_with_butterfly(model):
    return construct_effnet_for_dataset(
        model.efficientnet_type, model.dataset,
        pretrained=False, use_butterfly=True)


def get_device(device_name):
    assert device_name in ('cpu', 'cuda')
    return torch.device(device_name)

