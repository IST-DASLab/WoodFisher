"""
File implementing mixed convolution operations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy


class _Conv2dSamePadding(nn.Conv2d):
    """ Class implementing 2d adaptively padded convolutions """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, 0, dilation, groups, bias)
        # just for convenience in padding size computation
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)


# @deprecated uniform channel splits
def _get_split_sizes_old(channels, num_kernels):
        sizes = [channels // num_kernels] * num_kernels
        sizes[0] += channels % num_kernels
        return sizes 

# polynomial splits for channels
def _get_split_sizes(channels, num_kernels, p):
    sizes = []
    for i in range(1, num_kernels + 1):
        progress = min(float(i) / num_kernels , 1)
        remaining_progress = (1.0 - progress) ** p
        sizes.append(int(channels - channels * remaining_progress) 
                     - numpy.sum(sizes).astype(int))
    sizes[0] += channels - numpy.sum(sizes)
    return sizes

class Conv2dMixedSize(nn.Module):
    """
    Class that implements mixed 2d convolution
    It now support only the uniform channel groups
    and same **kwargs for all convs
    """
    def __init__(self, in_channels, out_channels, kernel_sizes, p, **kwargs):
        """
        Args:
          in_channels: number of input channels
          out_channels: number of output channels
          kernel_sizes: list of kernel size to consider in mixing
          kwargs: possibly containing {'stride': int_1, 
                                       'dilation': int_2, 
                                       'groups': int_3,
                                       'bias': bool_1}
        """
        super(Conv2dMixedSize, self).__init__()
        kernel_sizes = [9]

        self.p = p
        num_kernels = len(kernel_sizes)
        self.in_channels = _get_split_sizes(in_channels, num_kernels, self.p)
        self.out_channels = _get_split_sizes(out_channels, num_kernels, self.p)
        self.kernel_sizes = kernel_sizes
    
        for i in range(num_kernels):
            args = (self.in_channels[i], self.out_channels[i], self.kernel_sizes[i])
            setattr(self, f'conv{self.kernel_sizes[i]}x{self.kernel_sizes[i]}', 
                _Conv2dSamePadding(*args, **kwargs))

    def forward(self, x):
        # print(x.size())
        # print(self.conv3x3.weight.size())
        # print('==================================')
        assert x.size(1) == numpy.sum(self.in_channels), \
            'x number of channels does not much in_channels'
        outputs, chunks = [], torch.split(x, self.in_channels, dim=1)
        for i, chunk in enumerate(chunks):
            output = getattr(self, f'conv{self.kernel_sizes[i]}x{self.kernel_sizes[i]}')(chunk)
            outputs.append(output)
        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    print(_get_split_sizes(64,3))