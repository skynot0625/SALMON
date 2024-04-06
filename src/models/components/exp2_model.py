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
import torchvision.models as models
# 필요한 경우 추가 경로를 포함
sys.path.append('/path/to/src/aihwkit') 

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):

        def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
            super(SepConv, self).__init__()
            self.op = nn.Sequential(
                nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
                nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(channel_in, affine=affine),
                nn.ReLU(inplace=False),
                nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
                nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(channel_out, affine=affine),
                nn.ReLU(inplace=False),
            )

        def forward(self, x):
            return self.op(x)

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34"""

    # BasicBlock and Bottleneck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_conv=False):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion or use_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    @classmethod
    def from_digital(cls, digital_module, rpu_config, tile_module_class=None):
        # digital_module의 설정을 기반으로 아날로그 버전의 BasicBlock을 생성
        return cls(
            in_channels=digital_module.in_channels,
            out_channels=digital_module.out_channels,
            stride=digital_module.stride,
            use_conv=any(isinstance(layer, nn.Conv2d) for layer in digital_module.shortcut)
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class Bottleneck(nn.Module):
    """Residual block for resnet over 50 layers"""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_conv=False):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * Bottleneck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * Bottleneck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * Bottleneck.expansion or use_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNetBackbone(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, num_classes=10):
        super(ResNetBackbone, self).__init__()
        self.block = block
        self.layers = layers
        self.zero_init_residual = zero_init_residual
        self.groups = groups
        self.base_width = width_per_group
        if replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False, False, False]
        else:
            self.replace_stride_with_dilation = replace_stride_with_dilation
        self._norm_layer = norm_layer if norm_layer else nn.BatchNorm2d
        self.num_classes = num_classes
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=self.replace_stride_with_dilation[2])
        self.scala4 = nn.AvgPool2d(4, 4)
        self.fc4 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    @classmethod
    def from_digital(cls, digital_module, rpu_config, tile_module_class=None):
        # 이 메서드는 digital_module (디지털 버전의 ResNetBackbone)을 기반으로 아날로그 버전을 생성합니다.
        # 필요에 따라 이 메서드를 세부적으로 수정할 수 있습니다.
        return cls(
            block=digital_module.block,
            layers=digital_module.layers,
            zero_init_residual=digital_module.zero_init_residual,
            groups=digital_module.groups,
            width_per_group=digital_module.base_width,
            replace_stride_with_dilation=digital_module.replace_stride_with_dilation,
            norm_layer=digital_module._norm_layer,
            num_classes=10  # 이것은 고정되어 있습니다. 필요한 경우 변경하십시오.
        )
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        use_conv = stride != 1 or self.inplanes != planes * block.expansion
        layers.append(block(self.inplanes, planes, stride, use_conv))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)  # x1 output
        x2 = self.layer2(x1)  # x2 output
        x3 = self.layer3(x2)  # x3 output
        x4 = self.layer4(x3)
        
        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        if not hasattr(self, 'fc4') or self.fc4.in_features != out4_feature.size(1):
            self.fc4 = nn.Linear(out4_feature.size(1), 10).to(out4_feature.device)
            
        out4 = self.fc4(out4_feature)
        
        return out4, out4_feature, x4, x1, x2, x3


def create_resnet(architecture="resnet34", num_classes=10):
    """Create a ResNet-inspired model using the ResNetBackbone class.

    Args:
        architecture (str): Which ResNet architecture to create (options: "resnet18", "resnet34", "resnet50")
        num_classes (int): Number of output classes

    Returns:
        nn.Module: Created model
    """
    if architecture == "resnet18":
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif architecture == "resnet34":
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif architecture == "resnet50":
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif architecture == "resnet10":
        block = BasicBlock
        layers = [1, 1, 1, 1]
            
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    # Create the ResNetBackbone model
    model = ResNetBackbone(
        block=block,
        layers=layers,
        num_classes=num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=nn.BatchNorm2d,
    )

    return model

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

    # Create the 
class IntegratedResNet(nn.Module):
    def __init__(self, architecture="resnet10", num_classes=10, rpu_config=None):
        super(IntegratedResNet, self).__init__()
#         self.architecture = architecture
#         self.num_classes = num_classes
#         self.rpu_config = rpu_config
        self.backbone = create_resnet(architecture, num_classes)
        self.backbone = convert_to_analog(self.backbone, rpu_config=rpu_config)
        
    def forward(self, x):
        # ResNet 백본 통과
        out4, feature, x4, x1, x2, x3 = self.backbone(x)

        return out4, feature, x4

class IntegratedResNet_T(nn.Module):
    def __init__(self, architecture="resnet18", num_classes=10):
        super(IntegratedResNet_T, self).__init__()
        # 사전 훈련된 모델을 로드합니다.
        if architecture == "resnet18":
            self.backbone = models.resnet18(pretrained=True)
        elif architecture == "resnet34":
            self.backbone = models.resnet34(pretrained=True)
        elif architecture == "resnet50":
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

        # 사전 훈련된 모델의 마지막 FC 레이어를 제거합니다.
        self.backbone.fc = nn.Identity()
        
        # 새로운 FC 레이어를 클래스 초기화 시 정의합니다.
        # 주의: self.backbone.fc.in_features는 아직 알 수 없기 때문에, 임시 값을 사용합니다.
        self.new_fc = nn.Linear(512, num_classes)  # 512는 예시 값입니다. 실제 값은 아래에서 수정됩니다.

    def forward(self, x):
        # ResNet의 각 레이어를 통과합니다.
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x1 = self.backbone.layer1(x)
        x2 = self.backbone.layer2(x1)
        x3 = self.backbone.layer3(x2)
        x4 = self.backbone.layer4(x3)

        # 마지막 레이어의 출력을 평균 풀링하고 플래튼합니다.
        feature = self.backbone.avgpool(x4)
        feature = torch.flatten(feature, 1)

        # 새로운 FC 레이어의 in_features를 실제 feature 크기에 맞게 조정합니다.
        if self.new_fc.in_features != feature.shape[1]:
            self.new_fc = nn.Linear(feature.shape[1], self.new_fc.out_features).to(feature.device)
        
        out4 = self.new_fc(feature)

        return out4, feature, x4, x1, x2, x3