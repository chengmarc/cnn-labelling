# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
DATASET = "CIFAR100"
NET = "ResNet101"

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_TEST = 16

EPOCH = 200
LEARNING_RATE = 0.1
MILESTONES = [20, 40, 70, 110, 140, 160]
WARM = 1

USE_GPU = True
NUM_WORKERS = 4

