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

class SepConv_group_off(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv_group_off, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_out, channel_out, kernel_size=1, padding=0, bias=False),
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
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_conv=False):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion or use_conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        if not hasattr(self, 'fc4') or self.fc4.in_features != out4_feature.size(1):
            self.fc4 = nn.Linear(out4_feature.size(1), 10).to(out4_feature.device)
            
        out4 = self.fc4(out4_feature)
        
        return out4, out4_feature, x1, x2, x3

class ResNetAttention1(nn.Module):
    def __init__(self, block, num_classes=100):
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
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
            SepConv(channel_in=256 * block.expansion, channel_out=256 * block.expansion),
            nn.BatchNorm2d(256 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
            SepConv(channel_in=512 * block.expansion, channel_out=1024 * block.expansion),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

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
            SepConv(channel_in=512 * block.expansion, channel_out=512 * block.expansion),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=512 * block.expansion, channel_out=1024 * block.expansion),
            nn.AvgPool2d(4, 4)
        )
        
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

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

class ResNetInput(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, norm_layer=None):
        super(ResNetInput, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(base_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class ResNetFeatures(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None, base_channels=64):
        super(ResNetFeatures, self).__init__()
        self.inplanes = base_channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False, False, False]
        else:
            self.replace_stride_with_dilation = replace_stride_with_dilation

        self.groups = groups
        self.base_width = width_per_group
        self._norm_layer = norm_layer if norm_layer else nn.BatchNorm2d
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=self.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=self.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=self.replace_stride_with_dilation[2])
        self.scala4 = nn.AvgPool2d(4, 4)

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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        return out4_feature, x1, x2, x3

class ResNetClassifier(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x

def create_resnet_features(architecture="resnet34", zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
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

    return ResNetFeatures(block, layers, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)

def create_resnet_classifier(in_features, num_classes=10):
    return ResNetClassifier(in_features, num_classes)

def create_input_module(in_channels=3, base_channels=64, norm_layer=None):
    if norm_layer is None:
        norm_layer = nn.BatchNorm2d
    return ResNetInput(in_channels=in_channels, base_channels=base_channels, norm_layer=norm_layer)

class IntegratedResNet(nn.Module):
    def __init__(self, architecture="resnet10", num_classes=10):
        super(IntegratedResNet, self).__init__()
        self.input_module = create_input_module()
        self.features = create_resnet_features(architecture=architecture)
        
        block_type = BasicBlock if architecture in ["resnet18", "resnet34", "resnet10"] else Bottleneck
        in_features = 512 * block_type.expansion
        self.classifier = create_resnet_classifier(in_features=in_features, num_classes=num_classes)

        block_type_name = 'BasicBlock' if architecture in ["resnet18", "resnet34", "resnet10"] else 'Bottleneck'
        self.attention1 = ResNetAttention1(block_type_name, num_classes)
        self.attention2 = ResNetAttention2(block_type_name, num_classes)
        self.attention3 = ResNetAttention3(block_type_name, num_classes)

    def forward(self, x):
        x = self.input_module(x)
        out_features, x1, x2, x3 = self.features(x)
        
        out4 = self.classifier(out_features)
        
        out1, feature1 = self.attention1(x1)
        out2, feature2 = self.attention2(x2)
        out3, feature3 = self.attention3(x3)

        return out1, feature1, out2, feature2, out3, feature3, out4, out_features
