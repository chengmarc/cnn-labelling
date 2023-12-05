# -*- coding: utf-8 -*-
"""
@author: chengmarc
@github: https://github.com/chengmarc

[1] Yann LeCun, Patrick Haffner, Léon Bottou, and Yoshua Bengio.

    Object Recognition with Gradient Based Learning
    http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf

"""
import torch.nn as nn


# %% Class Definition

# ### Kernel Size / Stride / Padding ###
# Kernel size determine the size of the filter, which is usually an odd integer.
# - If the input size is a*b and the kernel size is 2n+1,
# - then the output size will be (a-2n)*(b-2n) assuming a,b greater than 2n.
# Stride determine how fast (how many pixel at a time) the filter will move horizontally and vertically.
# Padding determine how many pixels the original image should extend at the border.


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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


# %% Network Initializer


def initialize_network():

    return ConvNet()

