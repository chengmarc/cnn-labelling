# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader

import settings

from torch.optim.lr_scheduler import _LRScheduler

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



def train(epoch):

    start = time.time()
    net.train()
    
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if settings.USE_GPU:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()     
        
        trained_samples = batch_index * settings.BATCH_SIZE + len(images)
        total_samples = len(cifar100_training_loader.dataset)
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

    for (images, labels) in cifar100_test_loader:
        if settings.USE_GPU:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Evaluating Network.....')
    
    average_loss = '%0.4f'%(test_loss / len(cifar100_test_loader.dataset))
    accuracy = '%0.4f'%(correct.float() / len(cifar100_test_loader.dataset))
    print(f'[Test set]: Epoch: {epoch}, Average loss: {average_loss}, Accuracy: {accuracy}')

    finish = time.time()
    total = '%0.2f'%(finish-start)
    print(f'Epoch {epoch} testing time consumed: {total}s')
    print('')

    return correct.float() / len(cifar100_test_loader.dataset)

# %% Main Execution
if __name__ == '__main__':

    # Load Network
    import resnet    
    net, name = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2]), "ResNet18"
    if settings.USE_GPU: 
        net = net.cuda()
    
    # Load & Transform Data
    transform_train = transforms.Compose([transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15)])    
    transform_test = transforms.Compose([transforms.ToTensor()])
        
    cifar100_training = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
    cifar100_test = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    
    cifar100_training_loader = DataLoader(dataset=cifar100_training, shuffle=True, num_workers=2, batch_size=settings.BATCH_SIZE)
    cifar100_test_loader = DataLoader(dataset=cifar100_test, shuffle=True, num_workers=2, batch_size=settings.BATCH_SIZE)

    # Define Loss & Optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=settings.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    
    # Learning Rate Decay & Warm Up
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
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
            torch.save(net.state_dict(), f'./model/{settings.DATASET}-{name}-epoch-{epoch}-acc-{trunc_acc}.pt')
            best_acc = acc
    
    if settings.USE_GPU:
        print('')
        print(torch.cuda.memory_summary(), end='')