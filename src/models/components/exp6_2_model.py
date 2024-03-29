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

from aihwkit.simulator.configs import FloatingPointRPUConfig, FloatingPointDevice

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
        # self.residual_function의 결과에서 detach()를 호출하여
        # 이 부분의 그라디언트 업데이트를 차단
        residual = self.residual_function(x).detach()
        
        # shortcut 연산은 그대로 유지 (detach 호출 안 함)
        shortcut = self.shortcut(x)
        
        # 두 결과를 더한 뒤 ReLU 활성화 함수 적용
        return nn.ReLU(inplace=True)(residual + shortcut)



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
        # self.residual_function의 결과에서 detach()를 호출하여
        # 이 부분의 그라디언트 업데이트를 차단
        residual = self.residual_function(x).detach()
        
        # shortcut 연산은 그대로 유지 (detach 호출 안 함)
        shortcut = self.shortcut(x)
        
        # 두 결과를 더한 뒤 ReLU 활성화 함수 적용
        return nn.ReLU(inplace=True)(residual + shortcut)

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
        out4 = self.fc4(out4_feature)
        
        return out4, out4_feature, x1, x2, x3

class ResNetAttention1(nn.Module):
    def __init__(self, block, num_classes=100):  # num_classes 매개변수 추가
        super(ResNetAttention1, self).__init__()
        block = self._get_block_class(block)

        self.attention = nn.Sequential(
            SepConv(channel_in=64 * block.expansion, channel_out=64 * block.expansion),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=64 * block.expansion, channel_out=128 * block.expansion),
            SepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion),
            SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 출력 차원을 num_classes로 설정

    def forward(self, x1):
        fea = self.attention(x1)
        fea = fea * x1
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out
    
    @staticmethod
    def _get_block_class(block_name):
        if block_name == 'BasicBlock':
            return BasicBlock
        elif block_name == 'Bottleneck':
            return Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_name}")
    
class ResNetAttention2(nn.Module):
    def __init__(self, block, num_classes=100):
        super(ResNetAttention2, self).__init__()
        block = self._get_block_class(block)

        self.attention = nn.Sequential(
            SepConv(channel_in=128 * block.expansion, channel_out=128 * block.expansion),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion),
            SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x2):
        fea = self.attention(x2)
        fea = fea * x2
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out

    @staticmethod
    def _get_block_class(block_name):
        if block_name == 'BasicBlock':
            return BasicBlock
        elif block_name == 'Bottleneck':
            return Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_name}")
        
class ResNetAttention3(nn.Module):
    def __init__(self, block, num_classes=100):
        super(ResNetAttention3, self).__init__()
        block = self._get_block_class(block)
        
        self.attention = nn.Sequential(
            SepConv(channel_in=256 * block.expansion, channel_out=256 * block.expansion),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x3):
        fea = self.attention(x3)
        fea = fea * x3
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out
    
    @staticmethod
    def _get_block_class(block_name):
        if block_name == 'BasicBlock':
            return BasicBlock
        elif block_name == 'Bottleneck':
            return Bottleneck
        else:
            raise ValueError(f"Unknown block type: {block_name}")

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


def create_attention_resnets(num_classes=100, architecture="resnet34"):  # num_classes 매개변수 추가
    """Create ResNetAttention1, ResNetAttention2, and ResNetAttention3 models.

    Args:
        num_classes (int): Number of classes for the output layer. Default is 100.
        architecture (str): Which ResNet architecture to be used as a reference 
                            (options: "resnet18", "resnet34", "resnet50")

    Returns:
        tuple: Created ResNetAttention1, ResNetAttention2, ResNetAttention3 models
    """
    if architecture == "resnet18":
        block = BasicBlock
    elif architecture == "resnet34":
        block = BasicBlock
    elif architecture == "resnet50":
        block = Bottleneck
    elif architecture == "resnet10":
        block = BasicBlock
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    # Create the ResNetAttention models
    model1 = ResNetAttention1(block=block, num_classes=num_classes)  # num_classes 매개변수 전달
    model2 = ResNetAttention2(block=block, num_classes=num_classes)  # num_classes 매개변수 전달
    model3 = ResNetAttention3(block=block, num_classes=num_classes)  # num_classes 매개변수 전달

    return model1, model2, model3

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

class AnalogBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_conv=False, rpu_config=None):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            AnalogConv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AnalogConv2d(out_channels, out_channels * AnalogBasicBlock.expansion, kernel_size=3, padding=1, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(out_channels * AnalogBasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels * AnalogBasicBlock.expansion or use_conv:
            self.shortcut = nn.Sequential(
                AnalogConv2d(in_channels, out_channels * AnalogBasicBlock.expansion, kernel_size=1, stride=stride, bias=False, rpu_config=rpu_config),
                nn.BatchNorm2d(out_channels * AnalogBasicBlock.expansion)
            )

    @classmethod
    def from_digital(cls, digital_module, rpu_config, tile_module_class=None):
        return cls(
            in_channels=digital_module.in_channels,
            out_channels=digital_module.out_channels,
            stride=digital_module.stride,
            use_conv=any(isinstance(layer, nn.Conv2d) for layer in digital_module.shortcut),
            rpu_config=rpu_config
        )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class AnalogBottleneck(nn.Module):
    """Residual block for resnet over 50 layers using analog computation."""

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, use_conv=False, rpu_config=None):
        super(AnalogBottleneck, self).__init__()

        # residual function
        self.residual_function = AnalogSequential(
            AnalogConv2d(in_channels, out_channels, kernel_size=1, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AnalogConv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            AnalogConv2d(out_channels, out_channels * AnalogBottleneck.expansion, kernel_size=1, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(out_channels * AnalogBottleneck.expansion)
        )

        # shortcut
        self.shortcut = AnalogSequential()

        if stride != 1 or in_channels != out_channels * AnalogBottleneck.expansion or use_conv:
            self.shortcut = AnalogSequential(
                AnalogConv2d(in_channels, out_channels * AnalogBottleneck.expansion, kernel_size=1, stride=stride, bias=False, rpu_config=rpu_config),
                nn.BatchNorm2d(out_channels * AnalogBottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class AnalogResNetBackbone(nn.Module):
    def __init__(self, block, layers, rpu_config, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, num_classes=10):
        super(AnalogResNetBackbone, self).__init__()
        self.block = block
        self.layers = layers
        self.zero_init_residual = zero_init_residual
        self.groups = groups
        self.base_width = width_per_group
        self.replace_stride_with_dilation = replace_stride_with_dilation if replace_stride_with_dilation else [False, False, False]
        self._norm_layer = norm_layer if norm_layer else nn.BatchNorm2d
        self.num_classes = num_classes
        self.rpu_config = rpu_config
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = AnalogConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, rpu_config=rpu_config)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], rpu_config)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=self.replace_stride_with_dilation[0], rpu_config=rpu_config)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=self.replace_stride_with_dilation[1], rpu_config=rpu_config)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=self.replace_stride_with_dilation[2], rpu_config=rpu_config)
        self.scala4 = nn.AvgPool2d(4, 4)
        self.fc4 = AnalogLinear(512 * block.expansion, num_classes, rpu_config=rpu_config)

        for m in self.modules():
            if isinstance(m, AnalogConv2d):
                weight, bias = m.get_weights()
                nn.init.kaiming_normal_(weight, mode='fan_out', nonlinearity='relu')
                m.set_weights(weight, bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AnalogBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, AnalogBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    @classmethod
    def from_digital(cls, digital_module, rpu_config, tile_module_class=None):
        return cls(
            block=AnalogBasicBlock,  # 여기를 수정했습니다.
            layers=digital_module.layers,
            rpu_config=rpu_config,
            zero_init_residual=digital_module.zero_init_residual,
            groups=digital_module.groups,
            width_per_group=digital_module.base_width,
            replace_stride_with_dilation=digital_module.replace_stride_with_dilation,
            norm_layer=digital_module._norm_layer,
            num_classes=digital_module.num_classes
        )

    def _make_layer(self, block, planes, blocks, rpu_config, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = AnalogSequential(
                AnalogConv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, rpu_config=rpu_config),
                norm_layer(planes * block.expansion),
            )

        layers = []
        use_conv = stride != 1 or self.inplanes != planes * block.expansion
        layers.append(block(self.inplanes, planes, stride, use_conv, rpu_config=rpu_config))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, rpu_config=rpu_config))

        return AnalogSequential(*layers)

    # ... [Remaining parts of AnalogResNetBackbone remain the same] ...
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
            self.fc4 = AnalogLinear(out4_feature.size(1), 10, rpu_config=self.rpu_config).to(out4_feature.device)
            
        out4 = self.fc4(out4_feature)
        
        return out4, out4_feature, x1, x2, x3

def create_analog_resnet(architecture="resnet34", num_classes=10, rpu_config=None):
    """Create an Analog ResNet-inspired model using the AnalogResNetBackbone class.

    Args:
        architecture (str): Which ResNet architecture to create (options: "resnet18", "resnet34", "resnet50", "resnet10")
        num_classes (int): Number of output classes
        rpu_config: The configuration for the analog computation.

    Returns:
        nn.Module: Created analog model
    """
    if architecture == "resnet18":
        block = AnalogBasicBlock
        layers = [2, 2, 2, 2]
    elif architecture == "resnet34":
        block = AnalogBasicBlock
        layers = [3, 4, 6, 3]
    elif architecture == "resnet50":
        block = AnalogBottleneck
        layers = [3, 4, 6, 3]
    elif architecture == "resnet10":  # Added ResNet-10 support
        block = AnalogBasicBlock
        layers = [1, 1, 1, 1]
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    # Create the AnalogResNetBackbone model
    model = AnalogResNetBackbone(
        block=block,
        layers=layers,
        rpu_config=rpu_config,
        num_classes=num_classes,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=nn.BatchNorm2d,
    )

    return model

class AnalogSepConv(nn.Module):

    def __init__(self, channel_in, channel_out, rpu_config, kernel_size=3, stride=2, padding=1, affine=True):
        super(AnalogSepConv, self).__init__()
        self.op = nn.Sequential(
            AnalogConv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False, rpu_config=rpu_config),
            AnalogConv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            AnalogConv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False, rpu_config=rpu_config),
            AnalogConv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False, rpu_config=rpu_config),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

class AnalogResNetAttention1(nn.Module):
    def __init__(self, block, rpu_config, num_classes=100):
        super(AnalogResNetAttention1, self).__init__()
        
        self.attention = AnalogSequential(
            AnalogSepConv(channel_in=64 * block.expansion, channel_out=64 * block.expansion, rpu_config=rpu_config),
            nn.BatchNorm2d(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = AnalogSequential(
            AnalogSepConv(channel_in=64 * block.expansion, channel_out=128 * block.expansion, rpu_config=rpu_config),
            AnalogSepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion, rpu_config=rpu_config),
            AnalogSepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion, rpu_config=rpu_config),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = AnalogLinear(512 * block.expansion, num_classes, rpu_config=rpu_config)

    def forward(self, x1):
        fea = self.attention(x1)
        fea = fea * x1
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out
    
class AnalogResNetAttention2(nn.Module):
    def __init__(self, block, rpu_config, num_classes=100):
        super(AnalogResNetAttention2, self).__init__()
        
        self.attention = AnalogSequential(
            AnalogSepConv(channel_in=128 * block.expansion, channel_out=128 * block.expansion, rpu_config=rpu_config),
            nn.BatchNorm2d(128 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = AnalogSequential(
            AnalogSepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion, rpu_config=rpu_config),
            AnalogSepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion, rpu_config=rpu_config),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = AnalogLinear(512 * block.expansion, num_classes, rpu_config=rpu_config)

    def forward(self, x2):
        fea = self.attention(x2)
        fea = fea * x2
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out


class AnalogResNetAttention3(nn.Module):
    def __init__(self, block, rpu_config, num_classes=100):
        super(AnalogResNetAttention3, self).__init__()
        
        self.attention = AnalogSequential(
            AnalogSepConv(channel_in=256 * block.expansion, channel_out=256 * block.expansion, rpu_config=rpu_config),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = AnalogSequential(
            AnalogSepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion, rpu_config=rpu_config),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = AnalogLinear(512 * block.expansion, num_classes, rpu_config=rpu_config)

    def forward(self, x3):
        fea = self.attention(x3)
        fea = fea * x3
        feature_out = self.scala(fea).view(fea.size(0), -1)
        out = self.fc(feature_out)
        return out, feature_out

def create_analog_attention_resnets(rpu_config, num_classes=100, architecture="resnet34"):
    """Create AnalogResNetAttention1, AnalogResNetAttention2, and AnalogResNetAttention3 models.

    Args:
        rpu_config: Configuration for the analog layers.
        num_classes (int): Number of classes for the output layer. Default is 100.
        architecture (str): Which ResNet architecture to be used as a reference 
                            (options: "resnet18", "resnet34", "resnet50")

    Returns:
        tuple: Created AnalogResNetAttention1, AnalogResNetAttention2, AnalogResNetAttention3 models
    """
    if architecture == "resnet18":
        block = BasicBlock
    elif architecture == "resnet34":
        block = BasicBlock
    elif architecture == "resnet50":
        block = Bottleneck
    else:
        raise ValueError(f"Unknown architecture {architecture}")

    # Create the AnalogResNetAttention models
    model1 = AnalogResNetAttention1(block=block, rpu_config=rpu_config, num_classes=num_classes)
    model2 = AnalogResNetAttention2(block=block, rpu_config=rpu_config, num_classes=num_classes)
    model3 = AnalogResNetAttention3(block=block, rpu_config=rpu_config, num_classes=num_classes)

    return model1, model2, model3
    
class IntegratedResNet(nn.Module):
    def __init__(self, architecture="resnet10", num_classes=10, rpu_config=None):
        super(IntegratedResNet, self).__init__()
        self.backbone = create_resnet(architecture, num_classes)
        self.backbone = convert_to_analog(self.backbone, rpu_config)

        # 어텐션 모듈들 생성
        block_type = 'BasicBlock' if architecture in ["resnet18", "resnet34", "resnet10"] else 'Bottleneck'
        self.attention1 = ResNetAttention1(block_type, num_classes)
        self.attention2 = ResNetAttention2(block_type, num_classes)
        self.attention3 = ResNetAttention3(block_type, num_classes)

        # rpu_config_float = FloatingPointRPUConfig()
        # self.attention1 = convert_to_analog(self.attention1, rpu_config_float)
        # self.attention2 = convert_to_analog(self.attention2, rpu_config_float) 
        # self.attention3 = convert_to_analog(self.attention3, rpu_config_float)

    def forward(self, x):
        # ResNet 백본 통과
        out4, feature, x1, x2, x3 = self.backbone(x)

        # 어텐션 모듈들을 각각의 특성 맵에 적용
        out1, feature1 = self.attention1(x1)
        out2, feature2 = self.attention2(x2)
        out3, feature3 = self.attention3(x3)

        return out1, feature1, out2, feature2, out3, feature3, out4, feature