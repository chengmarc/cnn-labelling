# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Y. LeCun, and C. Cortes.

    MNIST handwritten digit database
    http://yann.lecun.com/exdb/mnist/

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

import settings


# %% EMNIST Training Data
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

data_train = MNIST(root='./data', train=True, download=True, transform=transform_train)
data_loader_train = DataLoader(dataset=data_train, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TRAIN)


# %% EMNIST Testing Data
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

data_test = MNIST(root='./data', train=False, download=True, transform=transform_test)
data_loader_test = DataLoader(dataset=data_test, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TEST)

