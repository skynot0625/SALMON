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
    @classmethod
    def from_digital(cls, digital_module, rpu_config, tile_module_class=None):
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
        return cls(
            block=digital_module.block,
            layers=digital_module.layers,
            zero_init_residual=digital_module.zero_init_residual,
            groups=digital_module.groups,
            width_per_group=digital_module.base_width,
            replace_stride_with_dilation=digital_module.replace_stride_with_dilation,
            norm_layer=digital_module._norm_layer,
            num_classes=10
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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        if not hasattr(self, 'fc4') or self.fc4.in_features != out4_feature.size(1):
            self.fc4 = nn.Linear(out4_feature.size(1), 10).to(out4_feature.device)
            
        out4 = self.fc4(out4_feature)
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

class ResNetAttention4(nn.Module):
    """Attention module for 512-channel feature maps (used for D6)."""
    
    def __init__(self, block, num_classes=100):
        super(ResNetAttention4, self).__init__()
        block = self._get_block_class(block)

        self.attention = nn.Sequential(
            SepConv(channel_in=512 * block.expansion, channel_out=512 * block.expansion),
            nn.BatchNorm2d(512 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )
        
        self.scala = nn.Sequential(
            SepConv(channel_in=512 * block.expansion, channel_out=512 * block.expansion),
            nn.AdaptiveAvgPool2d(1)  # Use adaptive pooling to handle variable input sizes
        )
        
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x4mid):
        fea = self.attention(x4mid)
        fea = fea * x4mid
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

class ResNetFeatures(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, 
                 width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
        super(ResNetFeatures, self).__init__()
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            self.replace_stride_with_dilation = [False, False, False]
        else:
            self.replace_stride_with_dilation = replace_stride_with_dilation

        self.groups = groups
        self.base_width = width_per_group
        self._norm_layer = norm_layer if norm_layer else nn.BatchNorm2d

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        
        return out4_feature, x1, x2, x3
    
    def forward_with_taps(self, x):
        """Forward with intermediate block taps for D4, D5, D6 attention heads.
        
        Returns:
            tuple: (out4_feature, x1, x2, x3, x2_mid, x3_mid, x4_mid)
                - out4_feature: final pooled feature
                - x1: layer1 output (for D1)
                - x2: layer2 output (for D2)
                - x3: layer3 output (for D3)
                - x2_mid: layer2[0] output (for D4)
                - x3_mid: layer3[0] output (for D5)
                - x4_mid: layer4[0] output (for D6)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # layer1 (2 blocks)
        x1 = self.layer1(x)

        # layer2 (2 blocks) - capture intermediate
        x2_mid = self.layer2[0](x1)   # tap for D4 (128-ch)
        x2 = self.layer2[1](x2_mid)

        # layer3 (2 blocks) - capture intermediate
        x3_mid = self.layer3[0](x2)   # tap for D5 (256-ch)
        x3 = self.layer3[1](x3_mid)

        # layer4 (2 blocks) - capture intermediate
        x4_mid = self.layer4[0](x3)   # tap for D6 (512-ch)
        x4 = self.layer4[1](x4_mid)

        out4_feature = self.scala4(x4).view(x4.size(0), -1)
        return out4_feature, x1, x2, x3, x2_mid, x3_mid, x4_mid

class ResNetClassifier(nn.Module):
    def __init__(self, in_features, num_classes=10):
        super(ResNetClassifier, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.fc(x)
        return x
    
def create_resnet_features(architecture="resnet34", zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
    """Create a ResNet Features model.

    Args:
        architecture (str): Which ResNet architecture to create (options: "resnet18", "resnet34", "resnet50", "resnet10")
        zero_init_residual (bool): Whether to zero-initialize the last BN in each residual branch.
        groups (int): Number of blocked connections from input channels to output channels.
        width_per_group (int): Width of each group.
        replace_stride_with_dilation (list of bool): Whether to replace the stride with dilation in some layers.
        norm_layer (nn.Module): Normalization layer used in the network.

    Returns:
        ResNetFeatures: The ResNet features extractor.
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

    return ResNetFeatures(block, layers, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)


def create_resnet_classifier(in_features, num_classes=10):
    """Create a ResNet Classifier model.

    Args:
        in_features (int): Number of input features for the classifier.
        num_classes (int): Number of output classes.

    Returns:
        ResNetClassifier: The ResNet classifier.
    """
    return ResNetClassifier(in_features, num_classes)

    
class IntegratedResNet(nn.Module):
    def __init__(self, architecture="resnet10", num_classes=10, rpu_config=None):
        super(IntegratedResNet, self).__init__()
        self.features = create_resnet_features(architecture=architecture)
        self.features = convert_to_analog(self.features, rpu_config=rpu_config)
        block_type_1 = BasicBlock if architecture in ["resnet18", "resnet34", "resnet10"] else Bottleneck
        in_features = 512 * block_type_1.expansion
        self.classifier = create_resnet_classifier(in_features=in_features, num_classes=num_classes)
        rpu_config_float = FloatingPointRPUConfig()
        self.classifier = convert_to_analog(self.classifier, rpu_config=rpu_config_float)

        # Original attention modules (D1, D2, D3)
        block_type = 'BasicBlock' if architecture in ["resnet18", "resnet34", "resnet10"] else 'Bottleneck'
        self.attention1 = ResNetAttention1(block_type, num_classes)  # D1 @ x1 (64-ch)
        self.attention2 = ResNetAttention2(block_type, num_classes)  # D2 @ x2 (128-ch)
        self.attention3 = ResNetAttention3(block_type, num_classes)  # D3 @ x3 (256-ch)
        
        # New attention heads (D4, D5, D6)
        self.attention4 = ResNetAttention2(block_type, num_classes)  # D4 @ x2_mid (128-ch)
        self.attention5 = ResNetAttention3(block_type, num_classes)  # D5 @ x3_mid (256-ch)
        self.attention6 = ResNetAttention4(block_type, num_classes)  # D6 @ x4_mid (512-ch)

    def forward(self, x):
        # Keep original forward for backward compatibility
        out_features, x1, x2, x3 = self.features(x)
        
        out4 = self.classifier(out_features)
        
        out1, feature1 = self.attention1(x1)
        out2, feature2 = self.attention2(x2)
        out3, feature3 = self.attention3(x3)

        return out1, feature1, out2, feature2, out3, feature3, out4, out_features
    
    def forward_all_heads(self, x):
        """Forward pass returning all 6 attention heads (D1-D6) plus backbone.
        
        Returns:
            tuple: (
                out_backbone, out4_feature,
                D1_out, D1_feat,
                D4_out, D4_feat,
                D2_out, D2_feat,
                D5_out, D5_feat,
                D3_out, D3_feat,
                D6_out, D6_feat,
            )
        """
        # Get all intermediate taps
        out4_feature, x1, x2, x3, x2_mid, x3_mid, x4_mid = self.features.forward_with_taps(x)
        
        # Backbone classifier
        out_backbone = self.classifier(out4_feature)
        
        # All 6 attention heads in interleaved order
        D1_out, D1_feat = self.attention1(x1)       # 64-ch
        D4_out, D4_feat = self.attention4(x2_mid)   # 128-ch (NEW)
        D2_out, D2_feat = self.attention2(x2)       # 128-ch
        D5_out, D5_feat = self.attention5(x3_mid)   # 256-ch (NEW)
        D3_out, D3_feat = self.attention3(x3)       # 256-ch
        D6_out, D6_feat = self.attention6(x4_mid)   # 512-ch (NEW)
        
        return (
            out_backbone, out4_feature,
            D1_out, D1_feat,
            D4_out, D4_feat,
            D2_out, D2_feat,
            D5_out, D5_feat,
            D3_out, D3_feat,
            D6_out, D6_feat,
        )


# Test code
if __name__ == "__main__":
    print("Testing IntegratedResNet with 6 attention heads...")
    
    # Create model
    model = IntegratedResNet(
        architecture="resnet18", 
        num_classes=10, 
        rpu_config=FloatingPointRPUConfig()
    )
    
    # Test input
    x = torch.randn(2, 3, 32, 32)
    
    # Test forward_all_heads
    outputs = model.forward_all_heads(x)
    
    print("\nOutput shapes from forward_all_heads:")
    print(f"Backbone logits: {outputs[0].shape}")  # [2, 10]
    print(f"Backbone features: {outputs[1].shape}")  # [2, 512]
    print(f"D1 logits: {outputs[2].shape}")  # [2, 10]
    print(f"D1 features: {outputs[3].shape}")  # [2, 512]
    print(f"D4 logits: {outputs[4].shape}")  # [2, 10]
    print(f"D4 features: {outputs[5].shape}")  # [2, 512]
    print(f"D2 logits: {outputs[6].shape}")  # [2, 10]
    print(f"D2 features: {outputs[7].shape}")  # [2, 512]
    print(f"D5 logits: {outputs[8].shape}")  # [2, 10]
    print(f"D5 features: {outputs[9].shape}")  # [2, 512]
    print(f"D3 logits: {outputs[10].shape}")  # [2, 10]
    print(f"D3 features: {outputs[11].shape}")  # [2, 512]
    print(f"D6 logits: {outputs[12].shape}")  # [2, 10]
    print(f"D6 features: {outputs[13].shape}")  # [2, 512]
    
    print("\nTest backward compatibility with original forward:")
    old_outputs = model.forward(x)
    print(f"Original forward returns {len(old_outputs)} outputs")
    
    print("\nAll tests passed!")