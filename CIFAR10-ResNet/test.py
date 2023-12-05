#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import settings
from torchvision.datasets import CIFAR10

if __name__ == '__main__':

    # Load Network
    import resnet    
    net, name = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2]), "ResNet18"
    if settings.USE_GPU: 
        net = net.cuda()
    
    # Load & Transform Data
    transform_test = transforms.Compose([transforms.ToTensor()])    
    cifar100_test = CIFAR10(root='./data', train=False, transform=transform_test, download=True)    
    cifar100_test_loader = DataLoader(dataset=cifar100_test, shuffle=True, num_workers=2, batch_size=16)
    
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
            for n_iter, (image, label) in enumerate(cifar100_test_loader):
                if settings.USE_GPU:
                    image = image.cuda()
                    label = label.cuda()

                output = net(image)
                _, pred = output.topk(5, 1, largest=True, sorted=True)

                label = label.view(label.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()
                           
                correct_1 += correct[:, :1].sum() #compute top 1
                correct_5 += correct[:, :5].sum() #compute top 5
                
                print(f"Iteration: [{n_iter + 1}/{len(cifar100_test_loader)}]")              
               
        print('')
        print("Top 1 error: ", (1 - correct_1 / len(cifar100_test_loader.dataset)).item)
        print("Top 5 error: ", (1 - correct_5 / len(cifar100_test_loader.dataset)).item)
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
        print('')
        
        if settings.USE_GPU:
            print('')
            print(torch.cuda.memory_summary(), end='')
    
    else:
        print('No pre-trained model found.')
        
    
