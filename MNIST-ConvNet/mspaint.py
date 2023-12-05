# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

"""
import os, random, subprocess
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import settings


# %% Visualization


def visualize(data):
    index, fig = 0, plt.figure(figsize=(5, 5))
    for i in [random.randint(0, len(data)-1) for j in range(25)]:
        index += 1
        fig.add_subplot(5, 5, index)
        plt.xticks([])
        plt.yticks([])

        image_slice = data[i][0].numpy()
        plt.imshow(image_slice[0])
    print("Visualizing 25 random samples, close the image to continue...")
    print("")
    plt.show()


from mnist import data_train
visualize(data_train)


# %% Load Model
from convnet import initialize_network
net = initialize_network()
if settings.USE_GPU:
    net = net.cuda()

model_list = [f'./model/{x}' for x in os.listdir('./model')]
if model_list:
    model_list.sort(key=lambda x: os.path.getmtime(x))

    print(f'Loading {model_list[-1]}...')
    print("")
    net.load_state_dict(torch.load(model_list[-1])) #load latest model
    net.eval()

else:
    print('No pre-trained model found, train a model first.')
    print("")

# %% Write & Recognizey
from mnist import transform_test as transform


def write_mspaint():

    image = Image.new('L', (28, 28), color=0)
    image_path = 'test_image.png'
    image.save(image_path)

    process = subprocess.Popen(['mspaint', image_path])
    process.wait()

    modified_image = Image.open(image_path)
    modified_image = modified_image.convert('L')
    modified_image = transform(modified_image).cuda()
    modified_image = modified_image.unsqueeze(0)

    with torch.no_grad():
        net.eval()
        output = net(modified_image)

    softmax_scores = F.softmax(output, dim=1)
    softmax_scores = softmax_scores.tolist()[0]
    softmax_scores = ["{:0>5.2f}%".format(score*100) for score in softmax_scores]

    print("Predicted character:")
    for i in range(0, 10):
        print(i, ":", softmax_scores[i], end="\t")
        if (i + 1) % 5 == 0: print("")

    print("")
    boolean = input("Do you want to write another character? (Y/N)")
    return boolean


boolean = 'Y'
print("Please write a character in MS Paint and save it.")

while boolean == 'Y' or boolean == 'y':
    boolean = write_mspaint()

