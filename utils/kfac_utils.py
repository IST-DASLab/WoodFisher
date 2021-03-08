import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

def try_contiguous(x):
    if not x.is_contiguous():
        x = x.contiguous()

    return x

def _extract_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = x.transpose_(1, 2).transpose_(2, 3).contiguous()
    x = x.view(
        x.size(0), x.size(1), x.size(2),
        x.size(3) * x.size(4) * x.size(5))
    return x


def _extract_channel_patches(x, kernel_size, stride, padding):
    """
    :param x: The input feature maps.  (batch_size, in_c, h, w)
    :param kernel_size: the kernel size of the conv filter (tuple of two elements)
    :param stride: the stride of conv operation  (tuple of two elements)
    :param padding: number of paddings. be a tuple of two elements
    :return: (batch_size, out_h, out_w, in_c*kh*kw)
    """
    if padding[0] + padding[1] > 0:
        x = F.pad(x, (padding[1], padding[1], padding[0],
                      padding[0])).data  # Actually check dims
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])  # b * oh * ow * kh * kw * inc
    x = x.transpose_(1, 2).transpose_(2, 3).transpose_(3, 4).transpose(4, 5).contiguous()
    x = x.view(x.size(0), x.size(1), x.size(2), x.size(3), x.size(4), x.size(5))
    return x


def update_running_stat(aa, m_aa, stat_decay):
    # using inplace operation to save memory!
    m_aa *= stat_decay / (1 - stat_decay)
    m_aa += aa
    m_aa *= (1 - stat_decay)


def fetch_mat_weights(layer, use_patch=False):
    # -> output_dium * input_dim (kh*kw*in_c + [1 if with bias])
    if isinstance(layer, nn.Conv2d):
        if use_patch:
            weight = layer.weight.transpose(1, 2).transpose(2, 3)  # n_out * kh * kw * inc
            n_out, k_h, k_w, in_c = weight.size()
            weight = try_contiguous(weight)
            weight = weight.view(-1, weight.size(-1))
            bias = 0
            if layer.bias is not None:
                copied_bias = torch.cat([layer.bias.unsqueeze(1) for _ in range(k_h*k_w)], 1).view(-1, 1)
                weight = torch.cat([weight, copied_bias], 1)  # layer.bias.unsqueeze(1)], 1)
                bias = 1
            weight = weight.view(n_out, k_h*k_w, in_c+bias)
        else:
            weight = layer.weight  # n_filters * in_c * kh * kw
            # weight = weight.transpose(1, 2).transpose(2, 3).contiguous()
            weight = weight.view(weight.size(0), -1)
            if layer.bias is not None:
                weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    elif isinstance(layer, nn.Linear):
        weight = layer.weight
        if layer.bias is not None:
            weight = torch.cat([weight, layer.bias.unsqueeze(1)], 1)
    else:
        raise NotImplementedError

    return weight


def mat_to_weight_and_bias(mat, layer):
    if isinstance(layer, nn.Conv2d):
        # mat: n_filters * (in_c * kh * kw)
        k_h, k_w = layer.kernel_size
        in_c = layer.in_channels
        out_c = layer.out_channels
        bias = None
        if layer.bias is not None:
            bias = mat[:, -1]
            mat = mat[:, :-1]
        weight = mat.view(out_c, in_c, k_h, k_w)
    elif isinstance(layer, nn.Linear):
        in_c = layer.in_features
        out_c = layer.out_features
        bias = None
        if layer.bias is not None:
            bias = mat[:, -1]
            mat = mat[:, :-1]
        weight = mat
    else:
        raise NotImplementedError
    return weight, bias

def rm_hooks(model):
    known_modules = {'WrappedLayer', 'Linear', 'Conv2d'}
    # known_modules = {'Linear', 'Conv2d'}
    for m in model.modules():
        classname = m.__class__.__name__
        if classname in known_modules:
            # if classname == 'WrappedLayer':
            #     m._backward_hooks = OrderedDict()
            #     m._layer._forward_pre_hooks = OrderedDict()
            # else:
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()
    return model

class ComputeMatGrad:

    @classmethod
    def __call__(cls, input, grad_output, layer):
        if isinstance(layer, nn.Linear):
            grad = cls.linear(input, grad_output, layer)
        elif isinstance(layer, nn.Conv2d):
            grad = cls.conv2d(input, grad_output, layer)
        else:
            raise NotImplementedError
        return grad

    @staticmethod
    def linear(input, grad_output, layer):
        """
        :param input: batch_size * input_dim
        :param grad_output: batch_size * output_dim
        :param layer: [nn.module] output_dim * input_dim
        :return: batch_size * output_dim * (input_dim + [1 if with bias])
        """
        with torch.no_grad():
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.unsqueeze(1)
            grad_output = grad_output.unsqueeze(2)
            grad = torch.bmm(grad_output, input)
        return grad

    @staticmethod
    def conv2d(input, grad_output, layer):
        """
        :param input: batch_size * in_c * in_h * in_w
        :param grad_output: batch_size * out_c * h * w
        :param layer: nn.module batch_size * out_c * (in_c*k_h*k_w + [1 if with bias])
        :return:
        """
        with torch.no_grad():
            input = _extract_patches(input, layer.kernel_size, layer.stride, layer.padding)
            input = input.view(-1, input.size(-1))  # b * hw * in_c*kh*kw
            grad_output = grad_output.transpose(1, 2).transpose(2, 3)
            grad_output = try_contiguous(grad_output).view(grad_output.size(0), -1, grad_output.size(-1))
            # b * hw * out_c
            if layer.bias is not None:
                input = torch.cat([input, input.new(input.size(0), 1).fill_(1)], 1)
            input = input.view(grad_output.size(0), -1, input.size(-1))  # b * hw * in_c*kh*kw
            grad = torch.einsum('abm,abn->amn', (grad_output, input))
        return grad


class ComputeCovA:
    # only tracks the blockwise (wrt layers) matrices for KFAC
    @classmethod
    def compute_cov_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        from utils.masking_utils import is_wrapped_layer
        if is_wrapped_layer(layer):
            if isinstance(layer._layer, nn.Linear):
                cov_a = cls.linear(a, layer._layer)
            elif isinstance(layer._layer, nn.Conv2d):
                cov_a = cls.conv2d(a, layer._layer)
            else:
                # raise NotImplementedError
                cov_a = None
        else:
            if isinstance(layer, nn.Linear):
                cov_a = cls.linear(a, layer)
            elif isinstance(layer, nn.Conv2d):
                cov_a = cls.conv2d(a, layer)
            else:
                # raise NotImplementedError
                cov_a = None

        return cov_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ (a / batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        print("before doing outer prods, shape of a is ", a.shape)
        return a.t() @ (a / batch_size)

class ComputeCovA_proper(ComputeCovA):
    # only tracks the blockwise (wrt layers) matrices for KFAC

    @staticmethod
    def conv2d(a, layer):
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a.t() @ a

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        # print('shape of a', a.shape)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        # print("before doing outer prods, shape of a is ", a.shape)
        return a.t() @ a


class ComputeCovG:
    # only tracks the blockwise (wrt layers) matrices for KFAC
    @classmethod
    def compute_cov_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        from utils.masking_utils import is_wrapped_layer
        if is_wrapped_layer(layer):
            if isinstance(layer._layer, nn.Linear):
                cov_g = cls.linear(g, layer._layer, batch_averaged)
            elif isinstance(layer._layer, nn.Conv2d):
                cov_g = cls.conv2d(g, layer._layer, batch_averaged)
            else:
                # raise NotImplementedError
                cov_g = None
        else:
            if isinstance(layer, nn.Conv2d):
                cov_g = cls.conv2d(g, layer, batch_averaged)
            elif isinstance(layer, nn.Linear):
                cov_g = cls.linear(g, layer, batch_averaged)
            else:
                cov_g = None

        return cov_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ (g / g.size(0))

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            cov_g = g.t() @ (g * batch_size)
        else:
            cov_g = g.t() @ (g / batch_size)
        return cov_g

class ComputeCovG_proper(ComputeCovG):

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
        g = g * spatial_size
        cov_g = g.t() @ g

        return cov_g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)
        # print('shape of g', g.shape)
        if batch_averaged:
            cov_g = g.t() @ (g * batch_size * batch_size)
        else:
            cov_g = g.t() @ g
        return cov_g


class StoreA:

    @classmethod
    def store_a(cls, a, layer):
        return cls.__call__(a, layer)

    @classmethod
    def __call__(cls, a, layer):
        if isinstance(layer, nn.Linear):
            store_a = cls.linear(a, layer)
        elif isinstance(layer, nn.Conv2d):
            store_a = cls.conv2d(a, layer)
        else:
            # raise NotImplementedError
            store_a = None

        return store_a

    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a / math.sqrt(batch_size)

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim
        batch_size = a.size(0)
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a / math.sqrt(batch_size)


class StoreG:

    @classmethod
    def store_g(cls, g, layer, batch_averaged=False):
        """
        :param g: gradient
        :param layer: the corresponding layer
        :param batch_averaged: if the gradient is already averaged with the batch size?
        :return:
        """
        # batch_size = g.size(0)
        return cls.__call__(g, layer, batch_averaged)

    @classmethod
    def __call__(cls, g, layer, batch_averaged):
        if isinstance(layer, nn.Conv2d):
            store_g = cls.conv2d(g, layer, batch_averaged)
        elif isinstance(layer, nn.Linear):
            store_g = cls.linear(g, layer, batch_averaged)
        else:
            store_g = None

        return store_g

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
            # this simply clears the normalization by the batch size!
        g = g * spatial_size
        return g / math.sqrt(g.size(0))

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            return g * math.sqrt(batch_size)
        else:
            return g / math.sqrt(batch_size)

class StoreA_proper(StoreA):

    @staticmethod
    def conv2d(a, layer):

        a = _extract_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        a = a/spatial_size
        return a

    @staticmethod
    def linear(a, layer):
        # a: batch_size * in_dim

        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1)], 1)
        return a


class StoreG_proper(StoreG):

    @staticmethod
    def conv2d(g, layer, batch_averaged):
        # g: batch_size * n_filters * out_h * out_w
        # n_filters is actually the output dimension (analogous to Linear layer)
        spatial_size = g.size(2) * g.size(3)
        batch_size = g.shape[0]
        g = g.transpose(1, 2).transpose(2, 3)
        g = try_contiguous(g)
        g = g.view(-1, g.size(-1))

        if batch_averaged:
            g = g * batch_size
            # this simply clears the normalization by the batch size!
        g = g * spatial_size
        return g

    @staticmethod
    def linear(g, layer, batch_averaged):
        # g: batch_size * out_dim
        batch_size = g.size(0)

        if batch_averaged:
            g = g * batch_size

        return g

class ComputeCovAPatch(ComputeCovA):
    @staticmethod
    def conv2d(a, layer):
        batch_size = a.size(0)
        a = _extract_channel_patches(a, layer.kernel_size, layer.stride, layer.padding)
        spatial_size = a.size(1) * a.size(2)
        a = a.view(-1, a.size(-1))
        patch_size = layer.kernel_size[0] * layer.kernel_size[1]
        if layer.bias is not None:
            a = torch.cat([a, a.new(a.size(0), 1).fill_(1./patch_size)], 1)
        a = a / spatial_size
        return a.t() @ (a / batch_size / patch_size)


if __name__ == '__main__':
    def test_ComputeCovA():
        pass

    def test_ComputeCovG():
        pass