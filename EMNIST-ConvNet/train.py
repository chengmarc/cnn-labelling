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

import settings
import shutup
shutup.please()


# %% Train & Evaluate Functions


def train(epoch):

    start = time.time()

    for batch_index, (images, labels) in enumerate(data_loader_train, 0):
        if settings.USE_GPU:
            images = images.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        scores = net(images)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        trained_samples = batch_index * settings.BATCH_SIZE_TRAIN + len(images)
        total_samples = len(data_loader_train.dataset)
        loss_rate = '%0.4f'%loss.item()
        learning_rate = '%0.6f'%optimizer.param_groups[0]['lr']
        print(f'Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {loss_rate}\tLR: {learning_rate}') 

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
        loss = criterion(outputs, labels)

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

    from convnet import initialize_network
    net = initialize_network()
    if settings.USE_GPU: 
        net = net.cuda()

    # Load Data
    from emnist import data_loader_train, data_loader_test

    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=settings.LEARNING_RATE, momentum=0.9)

    # FLY BITCH !!!
    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
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

