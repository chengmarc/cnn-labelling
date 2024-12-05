# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Baiyu @BUPT

    Practice on CIFAR100 using pytorch
    https://github.com/weiaicunzai/pytorch-cifar100

"""
import os
script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_path)

import torch

import settings
import shutup
shutup.please()


# %% Main Execution


if __name__ == '__main__':

    # Load Network
    from resnet import initialize_network
    net = initialize_network(settings.NET)
    if settings.USE_GPU: 
        net = net.cuda()

    # Load Data
    from cifar100 import data_loader_test

    # Check & Load Existing Weights
    model_list = [f'./model/{x}' for x in os.listdir('./model')]
    if model_list:
        model_list.sort(key=lambda x: os.path.getmtime(x))

        print(f'Loading {model_list[-1]}...')
        net.load_state_dict(torch.load(model_list[-1])) #load latest model
        net.eval()

        correct_1 = 0.0
        correct_5 = 0.0

        # Evaluation
        with torch.no_grad():
            for n_iter, (image, label) in enumerate(data_loader_test):
                if settings.USE_GPU:
                    image = image.cuda()
                    label = label.cuda()

                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                correct_1 += correct[:, :1].sum() #compute top 1
                correct_5 += correct[:, :5].sum() #compute top 5

                print(f"Iteration: [{n_iter + 1}/{len(data_loader_test)}]")              

        print('')
        print("Top 1 error: ", (1 - correct_1 / len(data_loader_test.dataset)).item())
        print("Top 5 error: ", (1 - correct_5 / len(data_loader_test.dataset)).item())
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
        print('')

        if settings.USE_GPU:
            print('')
            print(torch.cuda.memory_summary(), end='')

    else:
        print('No pre-trained model found, train a model first.')

