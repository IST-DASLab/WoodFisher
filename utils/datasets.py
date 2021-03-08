"""
Dataset loading utilities
"""

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import sklearn.datasets as sklearn_datasets

from torch.utils.data import TensorDataset

from utils.auto_augmentation import auto_augment_policy, AutoAugment

DATASETS_NAMES = ['imagenet', 'cifar10', 'cifar100', 'mnist', 'imagenette']

__all__ = ["get_datasets"]



def classification_dataset_str_from_arch(arch):
    if 'cifar100' in arch:
        dataset = 'cifar100'
    elif 'cifar' in arch:
        dataset = 'cifar10'
    elif 'mnist' in arch:
        dataset = 'mnist' 
    else:
        dataset = 'imagenet'
    return dataset


def classification_num_classes(dataset):
    return {'cifar10': 10,
            'mnist': 10,
            'imagenette': 10,
            'imagenet': 1000}.get(dataset, None)


def classification_get_input_shape(dataset):
    if dataset.startswith('imagenet'):
        return 1, 3, 224, 224
    elif dataset in ('cifar10', 'cifar100'):
        return 1, 3, 32, 32
    elif dataset == 'mnist':
        return 1, 1, 28, 28
    else:
        raise ValueError("dataset %s is not supported" % dataset)


def __dataset_factory(dataset):
    return globals()[f'{dataset}_get_datasets']


def get_datasets(dataset, dataset_dir, **kwargs):
    datasets_fn = __dataset_factory(dataset)
    if dataset == 'imagenet':
        return datasets_fn(dataset_dir, kwargs['use_aa'])
    elif dataset == 'cifar10':
        return datasets_fn(dataset_dir, kwargs['train_random_transforms'])
    return datasets_fn(dataset_dir)

def blobs_get_datasets(dataset_dir=None):
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')

    if os.path.isdir(dataset_dir):
        X_train, Y_train = torch.load(os.path.join(train_dir, 'x_data.pth')),\
                           torch.load(os.path.join(train_dir, 'y_data.pth'))
        X_test, Y_test = torch.load(os.path.join(test_dir, 'x_data.pth')),\
                         torch.load(os.path.join(test_dir, 'y_data.pth'))
    else:
        X, Y = sklearn_datasets.make_blobs(n_samples=15000,
                                           n_features=5,
                                           centers=3)
        X_train, Y_train = torch.FloatTensor(X[:-5000]), torch.FloatTensor(Y[:-5000])
        X_test, Y_test = torch.FloatTensor(X[-5000:]), torch.FloatTensor(Y[-5000:])
        
        # making dirs to save train/test
        os.makedirs(train_dir)
        os.makedirs(test_dir)

        torch.save(X_train, os.path.join(train_dir, 'x_data.pth'))
        torch.save(Y_train, os.path.join(train_dir, 'y_data.pth'))
        torch.save(X_test, os.path.join(test_dir, 'x_data.pth'))
        torch.save(Y_test, os.path.join(test_dir, 'y_data.pth'))

    # making torch datasets
    train_dataset = TensorDataset(X_train, Y_train.long())
    test_dataset = TensorDataset(X_test, Y_test.long())

    return train_dataset, test_dataset

def mnist_get_datasets(data_dir):
    # same used in hessian repo!
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                   download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                  transform=test_transform)

    return train_dataset, test_dataset


def cifar10_get_datasets(data_dir, train_random_transforms=True):
    if train_random_transforms:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def cifar100_get_datasets(data_dir):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True,
                                     download=True, transform=train_transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    test_dataset = datasets.CIFAR100(root=data_dir, train=False,
                                    download=True, transform=test_transform)

    return train_dataset, test_dataset


def imagenet_get_datasets(data_dir, use_aa=False):

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    train_transform = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]
    if use_aa:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        img_size_min = 224
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        aa_policy = auto_augment_policy('v0', aa_params)
        train_transform += [AutoAugment(aa_policy)]

    train_transform += [
        transforms.ToTensor(),
        normalize,
    ]
    train_transform = transforms.Compose(train_transform)

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset

def imagenette_get_datasets(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(train_dir, train_transform)

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(test_dir, test_transform)

    return train_dataset, test_dataset
