"""
Implementation of ResNets for cifar with regular convs replaced
by mixed kernel size convolutions (with dynamic same padding) 
"""
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

from models.layers import Conv2dMixedSize, drop_connect, SELayer

__all__ = ['resnet20_mixed']

NUM_CLASSES = 10
KERNEL_SIZES = [3,5,7]

def mixed_conv(in_planes, out_planes, stride=1):
    return Conv2dMixedSize(in_planes, out_planes, KERNEL_SIZES, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, block_gates, inplanes, planes, stride=1, 
                 downsample=None, drop_connect_rate=None):
        super(BasicBlock, self).__init__()
        self.block_gates = block_gates
        self.conv1 = mixed_conv(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=False)  # To enable layer removal inplace must be False
        self.conv2 = mixed_conv(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        residual = out = x

        if self.block_gates[0]:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)

        if self.block_gates[1]:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_connect_rate:
            out += drop_connect(residual, self.drop_connect_rate, self.training)
        else:
            out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, block_gates, inplanes, planes, stride=1, downsample=None,
                 drop_connect_rate=None, use_se=False, se_ratio=None):
        # dummy argument block_gates to support Bottleneck without bad ifs
        super(Bottleneck, self).__init__()

        self.use_se = use_se
        self.se_ratio = 0.5 if se_ratio is None else se_ratio
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2_1 = Conv2dMixedSize(planes, planes, KERNEL_SIZES, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.relu2_1 = nn.ReLU(inplace=True)

        self.conv2_2 = Conv2dMixedSize(planes, planes, KERNEL_SIZES, stride=stride, bias=False)
        self.bn2_2 = nn.BatchNorm2d(planes)
        self.relu2_2 = nn.ReLU(inplace=True)

        if self.use_se:
            self.se = SELayer(planes, self.se_ratio)
            self.se_bn = nn.BatchNorm2d(planes)
            self.se_relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu2_1(out)

        out = self.conv2_2(out)
        out = self.bn2_2(out)
        out = self.relu2_2(out)

        if self.use_se:
            out = self.se(out)
            out = self.se_bn(out)
            out = self.se_relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.drop_connect_rate:
            out += drop_connect(residual, self.drop_connect_rate, self.training)
        else:
            # print(out.size())
            # print(residual.size())
            out += residual
        out = self.relu3(out)

        return out


class ResNetCifar(nn.Module):

    def __init__(self, block, layers, drop_connect_rate=None, 
                 num_classes=NUM_CLASSES, use_se=False, se_ratio=None):
        self.nlayers = 0
        # Each layer manages its own gates
        self.layer_gates = []
        for layer in range(3):
            # For each of the 3 layers, create block gates: each block has two layers
            self.layer_gates.append([])  # [True, True] * layers[layer])
            for blk in range(layers[layer]):
                self.layer_gates[layer].append([True, True])

        self.use_se = use_se
        self.se_ratio = se_ratio

        self.drop_connect_rate = drop_connect_rate
        self.inplanes = 16  # 64
        super(ResNetCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(self.layer_gates[0], block, 16, layers[0])
        self.layer2 = self._make_layer(self.layer_gates[1], block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(self.layer_gates[2], block, 64, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, layer_gates, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(layer_gates[0], self.inplanes, planes, stride, downsample, self.drop_connect_rate, self.use_se, self.se_ratio))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(layer_gates[i], self.inplanes, planes, 
                                drop_connect_rate=self.drop_connect_rate, 
                                use_se=self.use_se, se_ratio=self.se_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet20(**kwargs):
    model = ResNetCifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet20_mixed(**kwargs):
    model = ResNetCifar(Bottleneck, [3,3,3], **kwargs)
    return model

def resnet32(**kwargs):
    model = ResNetCifar(BasicBlock, [5, 5, 5], **kwargs)
    return model

def resnet44(**kwargs):
    model = ResNetCifar(BasicBlock, [7, 7, 7], **kwargs)
    return model

def resnet56(**kwargs):
    model = ResNetCifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


if __name__ == '__main__':
    pass