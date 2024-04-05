import os
import sys
from datetime import datetime
import random
import numpy as np
import pandas as pd
import torch
from torch import nn, device, no_grad, manual_seed, save
from torch import max as torch_max
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.checkpoint as checkpoint
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.nn.conversion import convert_to_analog

# 필요한 경우 추가 경로를 포함
import torch.nn as nn

sys.path.append('/path/to/src/aihwkit') 

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        # CIFAR-10을 위한 수정된 classifier
        # 마지막 풀링 레이어 후 512 채널의 1x1 특성 맵이 예상됩니다.
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),  # 조정된 입력 크기
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)  # 특성 맵을 일렬로 펼침
        out = self.classifier(out)
        return out

    def _make_layers(self, config):
        layers = []
        in_channels = 3
        for x in config:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def create_vgg_network(model_name, n_classes):
    """Create a VGG network.

    Args:
        model_name (str): VGG model variant, 'VGG11' or 'VGG19'.
        n_classes (int): Number of classes for the output layer.

    Returns:
        nn.Module: Requested VGG model.
    """
    global N_CLASSES
    N_CLASSES = n_classes  # Update the global variable for the number of classes
    return VGG(model_name,n_classes)

class IntegratedResNet(nn.Module):
    def __init__(self, architecture="VGG11", num_classes=10, rpu_config=None):
        super(IntegratedResNet, self).__init__()
#         self.architecture = architecture
#         self.num_classes = num_classes
#         self.rpu_config = rpu_config
        self.backbone = create_vgg_network(model_name = architecture,  n_classes= num_classes)
        self.backbone = convert_to_analog(self.backbone, rpu_config=rpu_config)
        
    def forward(self, x):
        # 아날로그 백본 모델을 통과
        # 여기서는 예시로 VGG 모델의 출력을 사용합니다. ResNet의 경우 출력 형태를 확인하고 적절히 조정해야 합니다.
        out = self.backbone(x)
        # VGG 모델의 경우 마지막 레이어의 출력이 최종 출력입니다.
        # ResNet 모델의 경우, out 변수가 (out4, feature, x1, x2, x3)와 같은 튜플일 수 있으므로, 필요한 출력을 선택해야 합니다.
        return out
