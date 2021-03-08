"""
Impementation of squeeze and excitation operation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.mixed_conv import _Conv2dSamePadding
from models.layers.utils import swish, composite_swish



class SELayer(nn.Module):
    """
    Class impements the squeeze and excitation layer
    """
    def __init__(self, in_channels, se_ratio):
        super(SELayer, self).__init__()
        assert se_ratio <= 1.0 and se_ratio >= 0.,\
            'se_ratio should be in [0,1]'
        num_squeezed_channels = max(1, int(in_channels * se_ratio))
        self._se_reduce = _Conv2dSamePadding(in_channels, num_squeezed_channels, 1)
        self._se_expand = _Conv2dSamePadding(num_squeezed_channels, in_channels, 1)

    def forward(self, inputs):
        inputs_squeezed = F.adaptive_avg_pool2d(inputs, 1)
        inputs_squeezed = self._se_expand(swish(self._se_reduce(inputs_squeezed)))
        return composite_swish(inputs, inputs_squeezed)