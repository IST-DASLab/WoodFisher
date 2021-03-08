# -*- coding: utf-8 -*-
import math
from collections import OrderedDict

import torch
import torch.nn as nn


__all__ = ["resnet50_dpf"]


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding."
    return nn.Conv2d(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    [3 * 3, 64]
    [3 * 3, 64]
    """

    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    [1 * 1, x]
    [3 * 3, x]
    [1 * 1, x * 4]
    """

    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes, kernel_size=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)

        self.conv2 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.conv3 = nn.Conv2d(
            in_channels=out_planes,
            out_channels=out_planes * 4,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(num_features=out_planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        # assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class DownsampleC(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleC, self).__init__()
        assert stride != 1 or nIn != nOut
        self.conv = nn.Conv2d(
            nIn, nOut, kernel_size=1, stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DownsampleD(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleD, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ResNetBase(nn.Module):
    def _decide_num_classes(self):
        if self.dataset == "cifar10" or self.dataset == "svhn":
            return 10
        elif self.dataset == "cifar100":
            return 100
        elif self.dataset == "imagenet":
            return 1000

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _get_downsample_layer(self, block_fn, planes, stride=1, mode="conv_bn"):
        if mode == "conv_bn":
            downsample = DownsampleD(self.inplanes, planes * block_fn.expansion, stride)
        elif mode == "conv":
            downsample = DownsampleC(self.inplanes, planes * block_fn.expansion, stride)
        elif mode == "avg_pooling":
            downsample = DownsampleA(self.inplanes, planes * block_fn.expansion, stride)
        else:
            raise NotImplementedError("the avg sampling is not supported yet.")
        return downsample

    def _make_block(
        self, block_fn, planes, block_num, stride=1, downsample_mode="conv_bn"
    ):
        downsample = None
        if stride != 1 or self.inplanes != planes * block_fn.expansion:
            downsample = self._get_downsample_layer(
                block_fn=block_fn, planes=planes, stride=stride, mode=downsample_mode
            )

        layers = []
        layers.append(block_fn(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block_fn.expansion

        for _ in range(1, block_num):
            layers.append(block_fn(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResNet_imagenet(ResNetBase):
    def __init__(self, dataset, resnet_size, **kwargs):
        super(ResNet_imagenet, self).__init__()
        self.dataset = dataset

        # define model param.
        model_params = {
            18: {"block": BasicBlock, "layers": [2, 2, 2, 2]},
            34: {"block": BasicBlock, "layers": [3, 4, 6, 3]},
            50: {"block": Bottleneck, "layers": [3, 4, 6, 3]},
            101: {"block": Bottleneck, "layers": [3, 4, 23, 3]},
            152: {"block": Bottleneck, "layers": [3, 8, 36, 3]},
        }
        block_fn = model_params[resnet_size]["block"]
        block_nums = model_params[resnet_size]["layers"]

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_block(
            block_fn=block_fn, planes=64, block_num=block_nums[0]
        )
        self.layer2 = self._make_block(
            block_fn=block_fn, planes=128, block_num=block_nums[1], stride=2
        )
        self.layer3 = self._make_block(
            block_fn=block_fn, planes=256, block_num=block_nums[2], stride=2
        )
        self.layer4 = self._make_block(
            block_fn=block_fn, planes=512, block_num=block_nums[3], stride=2
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.linear = nn.Linear(
            in_features=512 * block_fn.expansion, out_features=self.num_classes
        )

        # weight initialization based on layer type.
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class ResNet_cifar(ResNetBase):
    def __init__(self, dataset, resnet_size, downsample_mode, **kwargs):
        super(ResNet_cifar, self).__init__()
        self.dataset = dataset

        # define model.
        if resnet_size % 6 != 2:
            raise ValueError("resnet_size must be 6n + 2:", resnet_size)
        block_nums = (resnet_size - 2) // 6
        block_fn = Bottleneck if resnet_size >= 44 else BasicBlock

        # decide the num of classes.
        self.num_classes = self._decide_num_classes()

        # define layers.
        self.inplanes = 16
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_block(
            block_fn=block_fn,
            planes=16,
            block_num=block_nums,
            downsample_mode=downsample_mode,
        )
        self.layer2 = self._make_block(
            block_fn=block_fn,
            planes=32,
            block_num=block_nums,
            stride=2,
            downsample_mode=downsample_mode,
        )
        self.layer3 = self._make_block(
            block_fn=block_fn,
            planes=64,
            block_num=block_nums,
            stride=2,
            downsample_mode=downsample_mode,
        )

        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.linear = nn.Linear(
            in_features=64 * block_fn.expansion, out_features=self.num_classes
        )

        # weight initialization based on layer type.
        self._weight_initialization()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def resnet(conf):
    """Constructs a ResNet-18 model.
    """
    resnet_size = int(conf.arch.replace("resnet", ""))
    dataset = conf.data

    if "cifar" in conf.data or "svhn" in conf.data:
        model = ResNet_cifar(
            dataset=dataset,
            resnet_size=resnet_size,
            downsample_mode=conf.resnet_downsample_mode,
        )
    elif "imagenet" in dataset:
        model = ResNet_imagenet(dataset=dataset, resnet_size=resnet_size)
    else:
        raise NotImplementedError
    return model


def resnet50_dpf(conf):
    """
       For comparison with Lin et. al. 2020, Dynamic Pruning with Feedback (DPF)
    """
    resnet_size = int(50)
    dataset = "imagenet"
    
    model = ResNet_imagenet(dataset=dataset, resnet_size=resnet_size)
    
    return model
