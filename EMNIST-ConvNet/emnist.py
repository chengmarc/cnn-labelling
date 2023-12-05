# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre van Schaik

    EMNIST: an extension of MNIST to handwritten letters
    https://arxiv.org/pdf/1702.05373v1.pdf
"""
from torch.utils.data import DataLoader
from torchvision.datasets import EMNIST
import torchvision.transforms as transforms

import settings


# %% EMNIST Training Data
transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

data_train = EMNIST(root='./data', split="bymerge", train=True, download=True, transform=transform_train)
data_loader_train = DataLoader(dataset=data_train, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TRAIN)


# %% EMNIST Testing Data
transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

data_test = EMNIST(root='./data', split="bymerge", train=False, download=True, transform=transform_test)
data_loader_test = DataLoader(dataset=data_test, shuffle=True, num_workers=settings.NUM_WORKERS, batch_size=settings.BATCH_SIZE_TEST)


# %% Label Dictionary
"""
By Merge: This data hierarchy addresses an interesting problem in the classification of handwritten digits, which
is the similarity between certain uppercase and lowercase letters. Indeed, these effects are often plainly visible when
examining the confusion matrix resulting from the full classification task on the By Class dataset. This variant
on the dataset merges certain classes, creating a 47-class classification task. The merged classes, as suggested by
the NIST, are for the letters C, I, J, K, L, M, O, P, S, U, V, W, X, Y and Z."

"""
dictionary = {0:"0  ", 1:"1  ", 2:"2  ", 3:"3  ", 4:"4  ", 5:"5  ", 6:"6  ", 7:"7  ", 8:"8  ", 9:"9  ",
              10:"A  ", 11:"B  ", 12:"C/c", 13:"D  ", 14:"E  ", 15:"F  ", 16:"G  ", 17:"H  ", 18:"I/i",
              19:"J/j", 20:"K/k", 21:"L/l", 22:"M/m", 23:"N  ", 24:"O/o", 25:"P/p", 26:"Q  ", 27:"R  ",
              28:"S/s", 29:"T  ", 30:"U/u", 31:"V/v", 32:"W/w", 33:"X/x", 34:"Y/y", 35:"Z/z", 36:"a  ",
              37:"b  ", 38:"d  ", 39:"e  ", 40:"f  ", 41:"g  ", 42:"h  ", 43:"n  ", 44:"q  ", 45:"r  ", 46:"t  "}

