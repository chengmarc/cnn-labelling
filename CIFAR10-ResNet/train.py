# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Baiyu @BUPT

    Practice on CIFAR100 using pytorch
    https://github.com/weiaicunzai/pytorch-cifar100

"""
import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

import settings
import shutup
shutup.please()


# %% Warm Up Learning Rate Scheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


# %% Train & Evaluate Functions


def train(epoch):

    start = time.time()
    net.train()

    for batch_index, (images, labels) in enumerate(data_loader_train):
        if settings.USE_GPU:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        trained_samples = batch_index * settings.BATCH_SIZE_TRAIN + len(images)
        total_samples = len(data_loader_train.dataset)
        loss_rate = '%0.4f'%loss.item()
        learning_rate = '%0.6f'%optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss_rate}\tLR: {learning_rate}')

        if epoch <= settings.WARM:
            warmup_scheduler.step()

    finish = time.time()
    total = '%0.2f'%(finish-start)
    print(f'Epoch {epoch} training time consumed: {total}s')
    print('')


@torch.no_grad()


def eval_training(epoch=0):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in data_loader_test:
        if settings.USE_GPU:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')

    average_loss = '%0.4f'%(test_loss / len(data_loader_test.dataset))
    accuracy = '%0.4f'%(correct.float() / len(data_loader_test.dataset))
    print(f'[Test set]: Epoch: {epoch}, Average loss: {average_loss}, Accuracy: {accuracy}')

    finish = time.time()
    total = '%0.2f'%(finish-start)
    print(f'Epoch {epoch} testing time consumed: {total}s')
    print('')

    return correct.float() / len(data_loader_test.dataset)


# %% Main Execution


if __name__ == '__main__':

    # Load Network
    from resnet import initialize_network
    net = initialize_network(settings.NET)
    if settings.USE_GPU: 
        net = net.cuda()

    # Load Data
    from cifar10 import data_loader_train, data_loader_test

    # Define Loss & Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

    # Learning Rate Decay & Warm Up
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(data_loader_train)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * settings.WARM)

    # FLY BITCH !!!
    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > settings.WARM:
            train_scheduler.step(epoch)

        train(epoch)
        acc = eval_training(epoch)

        if not os.path.exists('model'):
            os.makedirs('model')

        if best_acc < acc:
            trunc_acc = '%.4f'%acc.item()
            torch.save(net.state_dict(), f'./model/{settings.DATASET}-{settings.NET}-epoch-{epoch}-acc-{trunc_acc}.pt')
            best_acc = acc

    if settings.USE_GPU:
        print('')
        print(torch.cuda.memory_summary(), end='')

