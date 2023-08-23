# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import os
from PIL import Image
import subprocess
data_path = r"C:\Users\uzcheng\Desktop\repo\image-labelling\CNN-MNIST"

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

dataset_train = MNIST(root=data_path, train=True, transform=transform, download=False)
dataset_test = MNIST(root=data_path, train=True, transform=transform, download=False)

loader_train = DataLoader(dataset=dataset_train, batch_size=64, shuffle=True)
loader_test = DataLoader(dataset=dataset_test, batch_size=64, shuffle=False)

# %%
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2, 2)
        x = x.view(-1, 32 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# %%
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(loader_train, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
running_loss = 0.0

print('Finished training')

# %%

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(num_correct/num_samples)
        
check_accuracy(loader_test, model)

# %% Model prediction
boolean = 'Y'
print("Please write a number in MS Paint.")

while boolean == 'Y' or boolean == 'y':
    
    image = Image.new('L', (28, 28), color=0)
    image_path = 'temp_image.png'
    image.save(image_path)

    process = subprocess.Popen(['mspaint', image_path])
    process.wait()

    modified_image = Image.open(image_path)
    modified_image = modified_image.convert('L')
    modified_image = transform(modified_image)
    modified_image = modified_image.unsqueeze(0)
    
    with torch.no_grad():
        model.eval()
        output = model(modified_image)
        
    softmax_scores = F.softmax(output, dim=1)
    softmax_scores = softmax_scores.tolist()[0]
    softmax_scores = ["{:0>5.2f}%".format(score*100) for score in softmax_scores]
    
    for i in range(0, 10):
        print(i, ":", softmax_scores[i])
    print("")
    boolean = input("Do you want to write another number? (Y/N)")
    """
    
    
    modified_image = np.array(modified_image)
    modified_image = modified_image.reshape(-1, 28, 28, 1)

    import os
    os.remove(image_path)
    
    with tf.device('/CPU:0'): predictions = model.predict(modified_image)
    index = np.argmax(predictions)
    value = predictions[0, index]

    print('According to the prediction of the trained model,')
    print('the given number is', index, 'with a', value*100, '% confidence.')
    print()
    
    boolean = input("Do you want to write another number? (Y/N)") """