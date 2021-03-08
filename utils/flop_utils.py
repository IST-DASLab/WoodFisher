import torch
from utils.checkpoints import get_unwrapped_model

def get_macs(args, model):
    from torchprofile import profile_macs
    from utils.datasets import classification_get_input_shape

    inputs = torch.randn(classification_get_input_shape(args.dset))
    macs = profile_macs(model, inputs)
    return macs

def get_macs_dpf(args, model, multiply_adds=False, ignore_zero=True, display_log=True,
                 ignore_bn=False, ignore_relu=False, ignore_maxpool=False, ignore_bias=False):

    # Inspired from DPF code (Lin et al 2020)
    # ---------------
    # Code from https://github.com/simochen/model-tools.

    import numpy as np

    import torch.nn as nn

    """for cv tasks."""

    data = args.dset
    device = args.device

    if "cifar" in data:
        input_res = [3, 32, 32]
    elif "imagenet" in data:
        input_res = [3, 224, 224]
    elif "mnist" in data:
        input_res = [1, 28, 28]
    else:
        raise RuntimeError("not supported imagenet type.")

    prods = {}

    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)

        return hook_per

    list_1 = []

    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2 = {}

    def simple_hook2(self, input, output):
        list_2["names"] = np.prod(input[0].shape)

    list_conv = []
    module_names = []

    def conv_hook(self, input, output):
        # print(self.weight.shape)
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = (
                self.kernel_size[0] * self.kernel_size[1] *
                (self.in_channels / self.groups)
        )
        bias_ops = 1 if not ignore_bias and self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        assert self.weight.numel() == kernel_ops * output_channels, "Not match"
        flops = (
                (
                        num_weight_params * (2 if multiply_adds else 1)
                        + bias_ops * output_channels
                )
                * output_height
                * output_width
                * batch_size
        )

        list_conv.append(flops)
        module_names.append(self.name)

    list_linear = []


    def linear_hook(self, input, output):
        # print(self.weight.shape)
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        num_weight_params = (
            (self.weight.data != 0).float().sum()
            if ignore_zero
            else self.weight.data.nelement()
        )
        weight_ops = num_weight_params * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement() if not ignore_bias else 0

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)
        module_names.append(self.name)

    list_bn = []

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu = []

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling = []

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (
                (kernel_ops + bias_ops)
                * output_channels
                * output_height
                * output_width
                * batch_size
        )

        list_pooling.append(flops)

    list_upsample = []

    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)


    def foo(net, name=''):

        children = list(net.named_children())
        if not children:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
                setattr(net, 'name', name)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(
                    net, torch.nn.AvgPool2d
            ):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for child_name, child in children:
            foo(child, name="{}.{}".format(name, child_name))

    assert model is not None
    # print(model)
    foo(model)
    # 1, 3, 224, 224
    _input = torch.rand(*input_res).unsqueeze(0).to(device)

    model(_input)

    total_flops = (
            sum(list_conv)
            + sum(list_linear)
            + (sum(list_bn) if not ignore_bn else 0)
            + (sum(list_relu) if not ignore_relu else 0)
            + (sum(list_pooling) if not ignore_maxpool else 0)
            + sum(list_upsample)
    )
    total_flops = (
        total_flops.item() if isinstance(total_flops, torch.Tensor) else total_flops
    )
    list_conv = [x.item() for x in list_conv]
    list_linear = [x.item() for x in list_linear]
    print("list conv is ", list_conv)
    print("list linear is ", list_linear)
    print("list module_names is ", module_names)

    print(sum(list_linear) + sum(list_conv))
    if display_log:
        print(
            "  + Number of {}: {:.3f}M".format(
                "flop" if multiply_adds else "macs", 1.0 * total_flops / 1e6
            )
        )
    return total_flops, list_conv + list_linear, module_names

def get_flops(args, model):
    model = get_unwrapped_model(model)
    model = get_unwrapped_model(model)
    total_flops, module_flops, module_names = get_macs_dpf(args, model, multiply_adds=False, ignore_zero=True,
                                                           display_log=True,
                                                           ignore_bn=True, ignore_relu=True, ignore_maxpool=True,
                                                           ignore_bias=True)

    try:
        from prettytable import PrettyTable

        tab = PrettyTable()

        tab.field_names = ["Module Name", "FLOPs (M)"]

        assert len(module_flops) == len(module_names)
        for i in range(len(module_flops)):
            tab.add_row([module_names[i], 1.0 * module_flops[i] / 1e6])
        tab.add_row(['Total', 1.0 * total_flops / 1e6])

        print(tab.get_string(title="FLOP information"))

    except ImportError:
        print('Install prettytable by `pip install PTable` for a prettier log of FLOP information')

    return total_flops, module_flops, module_names
