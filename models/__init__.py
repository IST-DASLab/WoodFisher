"""
Example models to train and prune.
Interface is provided by the get_model function.
"""
import torch

from models.resnet_imagenet import *
from models.resnet_cifar10 import *
from models.resnet_cifar10_swish import *
from models.logistic_regression import *
from models.resnet_mixed_cifar10 import *
from models.resnet_mixed_imagenet import *
from models.wide_resnet_imagenet import *
from models.mobilenet import *
from models.mobilenet_v1_dropout import * 
from models.mobilenetv2 import *
from models.resnet_dpf import *
from models.mlpnet import *
from models.cifarnet import *

from torchvision.models import resnet50 as torch_resnet50
from torchvision.models import vgg16_bn, vgg19
from torchvision.models import vgg11, vgg11_bn, inception_v3

CIFAR10_MODELS = ['resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet20_sw', 'resnet32_sw', 'resnet44_sw', 'resnet56_sw', 'resnet20_mixed', 'cifarnet']
IMAGENET_MODELS = ['resnet18', 'resnet34', 'vgg19', 'resnet50', 'resnet101', 'resnet152', 'resnet50_mixed',
    'wide_resnet50_2_mixed', 'mobilenet', 'mobilenet_v2', 'resnet_dpf', 'inception_v3', 'mobilenet_v1_dropout']

def get_model(name, dataset, pretrained=False, use_butterfly=False,
              use_se=False, se_ratio=None, kernel_sizes=None, p=None, args=None):
    if name.startswith('resnet') and dataset == 'cifar10':
        if name == 'resnet50' and pretrained:
            return torch_resnet50(pretrained=True)
        try:
            if 'mixed' in name:
                assert_use_se(name, use_se)
                return globals()[name](**{'use_se':use_se, 'se_ratio': se_ratio})
            return globals()[name]()
        except:
            raise ValueError(f'Model {name} is not supported for {dataset}, list of supported: {", ".join(CIFAR10_MODELS)}')
    if 'resnet' in name and any([dataset == 'imagenet', dataset == 'imagenette']):
        if 'mixed' in name:
            assert_use_se(name, use_se)
            kwargs_dict = {'use_se':use_se, 'se_ratio': se_ratio, 
                           'kernel_sizes': kernel_sizes, 'p': p}
            if dataset == 'imagenette':
                kwargs_dict['num_classes'] = 10
            return globals()[name](**kwargs_dict)
        return globals()[name](pretrained)
        try:
            if 'mixed' in name:
                assert_use_se(name, use_se)
                kwargs_dict = {'use_se':use_se, 'se_ratio': se_ratio, 
                               'kernel_sizes': kernel_sizes, 'p': p}
                if dataset == 'imagenette':
                    kwargs_dict['num_classes'] = 10
                return globals()[name](**kwargs_dict)
            return globals()[name](pretrained)
        except:
            raise ValueError(f'Model {name} is not supported for {dataset}, list of supported: {", ".join(IMAGENET_MODELS)}')
    if name == 'cifarnet':
        return CIFARNet()
    if name == 'mlpnet':
        return MlpNet(args, dataset)
    if name == 'logistic_regression' and dataset == 'blobs':
        return LogisticRegression()
    if 'mobilenet' in name or 'inception' in name:
        print ("loading " + name)
        return globals()[name](pretrained)
    return globals()[name](pretrained)   
    raise NotImplementedError

def assert_use_se(name, use_se):
    if any([name == 'resnet20_mixed', name == 'resnet50_mixed', 
        name == 'wide_resnet50_2_mixed']) and use_se:
        return
    elif use_se:
        raise NotImplementedError("SELayer is only implemented for mixed resnet20 model")

if __name__ == '__main__':
    get_model('resnet', 'cifar10', pretrained=False)
