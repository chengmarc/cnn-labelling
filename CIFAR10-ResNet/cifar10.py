# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Krizhevsky, A., Nair, V. and Hinton, G.

    The CIFAR-10 Dataset
    https://www.cs.toronto.edu/~kriz/cifar.html

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

import settings


# %% CIFAR10 Training Data
transform_train = transforms.Compose([transforms.ToTensor(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15)])

data_train = CIFAR10(root='./data', train=True, download=True, transform=transform_train)
data_loader_train = DataLoader(dataset=data_train, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TRAIN)


# %% CIFAR10 Testing Data
transform_test = transforms.Compose([transforms.ToTensor()])

data_test = CIFAR10(root='./data', train=False, download=True, transform=transform_test)
data_loader_test = DataLoader(dataset=data_test, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TEST)

