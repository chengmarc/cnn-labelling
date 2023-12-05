""" configurations for this project

author baiyu
"""
#total training epoches

DATASET = "CIFAR10"

EPOCH = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MILESTONES = [60, 120, 160]
WARM = 1

USE_GPU = True