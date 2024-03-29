# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023 IBM. All Rights Reserved.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""aihwkit example 11: analog CNN.

SVHN dataset on Analog Network using weight scaling.

Learning rates of η = 0.1 for all the epochs with minibatch 128.
"""
# pylint: disable=invalid-name

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Imports from aihwkit.
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.simulator.rpu_base import cuda



import torch.nn as nn
# Assuming analog layers are defined in a specific module for analog neural network components
from analog_layers import AnalogSequential, AnalogConv2d, AnalogLinear, RPU_CONFIG

N_CLASSES = 10  # Define the number of output classes here

class AnalogVGG8Net(nn.Module):
    def __init__(self, num_classes=N_CLASSES, rpu_config=RPU_CONFIG):
        super(AnalogVGG8Net, self).__init__()
        channel_base = 48
        channels = [channel_base, 2 * channel_base, 3 * channel_base]
        fc_size = 8 * channel_base
        self.features = AnalogSequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            AnalogConv2d(channels[0], channels[0], kernel_size=3, padding=1, rpu_config=rpu_config),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers following the pattern above
        )
        self.classifier = AnalogSequential(
            AnalogLinear(16 * channels[2], fc_size, rpu_config=rpu_config),
            nn.ReLU(),
            AnalogLinear(fc_size, num_classes, rpu_config=rpu_config),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class DigitalVGG8Net(nn.Module):
    def __init__(self, num_classes=N_CLASSES):
        super(DigitalVGG8Net, self).__init__()
        channel_base = 48
        channels = [channel_base, 2 * channel_base, 3 * channel_base]
        fc_size = 8 * channel_base
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Add more layers following the pattern above
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * channels[2], fc_size),
            nn.ReLU(),
            nn.Linear(fc_size, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
