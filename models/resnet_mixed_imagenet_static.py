import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import math

from models.layers import drop_connect, SELayer, Conv2dMixedSizeStatic



__all__ = ['resnet50_mixed_static' , 'get_resnet50_mixed_onnx_version']

PADS = [(2, 2),(4, 4),(6, 6),(8, 8),(2, 2),(4, 4),(6, 6),(8, 8),(2, 2),(4, 4),(6, 6),
        (8, 8),(1, 1),(3, 3),(5, 5),(7, 7),(2, 2),(4, 4),(6, 6),(8, 8),(2, 2),(4, 4),
        (6, 6),(8, 8),(2, 2),(4, 4),(6, 6),(8, 8),(1, 1),(3, 3),(5, 5),(7, 7),(2, 2),
        (4, 4),(6, 6),(2, 2),(4, 4),(6, 6),(2, 2),(4, 4),(6, 6),(2, 2),(4, 4),(6, 6),
        (2, 2),(4, 4),(6, 6),(1, 1),(3, 3),(5, 5),(2, 2),(4, 4),(2, 2),(4, 4)]

COUNTER = 0


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 drop_connect_rate=None, use_se=False, se_ratio=None, 
                 p=None, kernel_sizes=None):
        super(Bottleneck, self).__init__()

        global COUNTER

        self.use_se = use_se
        self.se_ratio = 0.5 if se_ratio is None else se_ratio

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2_1 = Conv2dMixedSizeStatic(planes, planes, 
            kernel_sizes, p, PADS[COUNTER:COUNTER+len(kernel_sizes)],  
            stride=stride, bias=False)
        self.bn2_1 = nn.BatchNorm2d(planes)
        self.relu2_1 = nn.ReLU(inplace=True)

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

        COUNTER += len(kernel_sizes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2_1(out)
        out = self.bn2_1(out)
        out = self.relu2_1(out)

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
            out += residual
        out = self.relu3(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, drop_connect_rate=None, 
                 use_se=False, se_ratio=None, p=None, kernel_sizes=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.drop_connect_rate = drop_connect_rate

        self.p = p
        self.kernel_sizes = kernel_sizes
        self.use_se = use_se
        self.se_ratio = se_ratio

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], kernel_sizes=[3,5,7,9])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, kernel_sizes=[3,5,7,9])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, kernel_sizes=[3,5,7])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, kernel_sizes=[3,5])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel_sizes=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.drop_connect_rate,
            self.use_se, self.se_ratio, self.p, kernel_sizes if len(kernel_sizes) == 4 else kernel_sizes + [kernel_sizes[-1] + 2]))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_connect_rate=self.drop_connect_rate,
                use_se=self.use_se, se_ratio=self.se_ratio, p=self.p, 
                kernel_sizes=kernel_sizes))

        return nn.Sequential(*layers)

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
        x = self.fc(x)

        return x


def resnet50_mixed_static(**kwargs):
    kwargs['kernel_sizes'] =[2 * i + 1 for i in range(1,kwargs['kernel_sizes'] + 1)] 
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    global COUNTER
    COUNTER = 0
    return model


def get_resnet50_mixed_onnx_version(ckpt_path, onnx_path):
    model = resnet50_mixed_static(**{'kernel_sizes': 4, 'p': 1})
    ckpt = torch.load(ckpt_path)

    def _normalize_module_name(layer_name):
        modules = layer_name.split('.')
        try:
            idx = modules.index('module')
            del modules[idx]
        except ValueError:
            pass
        try:
            idx = modules.index('_layer')
            del modules[idx]
        except ValueError:
            pass
        return '.'.join(modules)

    new_dict = dict()
    for key,val in ckpt['model_state_dict'].items():
        if '_mask' not in key:
            new_dict[_normalize_module_name(key)] = val

    dummy_input = torch.zeros(10,3,224,224)
    torch.onnx.export(model, dummy_input, onnx_path)


if __name__ == '__main__':
    pass