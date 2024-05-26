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
from aihwkit.simulator.presets import GokmenVlasovPreset,IdealizedPreset
from aihwkit.simulator.configs import (
    SingleRPUConfig,
    FloatingPointRPUConfig,
    ConstantStepDevice,
    FloatingPointDevice,
    IdealDevice
)

# 필요한 경우 추가 경로를 포함
sys.path.append('/path/to/src/aihwkit') 
# RPU_CONFIG = FloatingPointRPUConfig(device=FloatingPointDevice())
RPU_CONFIG = SingleRPUConfig(device=IdealDevice())
class Boxcar(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh, subthresh):
        # spike threshold, Heaviside
        # store membrane potential before reset
        ctx.save_for_backward(input)
        ctx.thresh = thresh
        ctx.subthresh = subthresh
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        # surrogate-gradient, BoxCar
        # stored membrane potential before reset
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.subthresh
        # return grad_input, None, None
        # return grad_input * temp.float(), None, None
        # return grad_input * temp.float()/(ctx.subthresh), None, None
        return grad_input * temp.float(), None, None


class HeavisideBoxcarCall(nn.Module):
    def __init__(self, thresh=1.0, subthresh=0.5, alpha=1.0, spiking=True):
        super().__init__()
        self.alpha = alpha
        self.spiking = spiking
        self.thresh = torch.tensor(thresh)
        self.subthresh = torch.tensor(subthresh)
        self.thresh.to("cuda" if torch.cuda.is_available() else "cpu")
        self.subthresh.to("cuda" if torch.cuda.is_available() else "cpu")
        if spiking:
            self.f = Boxcar.apply
        else:
            self.f = self.primitive_function

    def forward(self, x):
        return self.f(x, self.thresh, self.subthresh)

    @staticmethod
    def primitive_function(x: torch.Tensor, alpha):
        return (x * alpha)

class LIF_Node(nn.Module):
    def __init__(self, surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=1.0, alpha=1.0, spiking=True)):
        super().__init__()
        self.surrogate_function = surrogate_function

    def forward(self, mem: torch.Tensor, spike_before: torch.Tensor, decay: torch.Tensor, I_in: torch.Tensor):
        mem = mem * decay * (1-spike_before) + I_in
        spike = self.surrogate_function(mem)
        return mem, spike

class Spiking_ResNet18_LIF_STBP(nn.Module):
    def __init__(
        self,
        num_steps: int = 5,                 # or 10
        init_tau: float = 0.9,              # 0.9, membrane decaying time constant
        scale = 64, 
        subthresh  = 0.5,
        affine = True,
        kernel_size =3,
        stride = 2,
        padding = 1
    ):
        super().__init__()
        self.num_steps = num_steps
        self.scale = scale
        self.subthresh = subthresh
        '''
        def Spiking_ResNet11_Lee(num_class, snn_params, init_channels=128):
        c = init_channels
        model_spec = {
            'C_stem': c,
            'channels': [c, c, c, c*2, c*2, c*2, c*4, c*4],
            'C_last': c*2, ## FC1,
            'strides': [1, 1, 1, 2, 1, 1, 2, 1],
            'use_downsample_avg': False,
            'last_avg_pool': '2x2',
        }
        '''
        self.conv1 = AnalogConv2d(3, self.scale, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)   
        self.bn1 = nn.BatchNorm2d(self.scale)
        self.LIF1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SRB1_conv1 = AnalogConv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        
        self.SRB1_bn1 = nn.BatchNorm2d(self.scale)
        self.LIF1_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_conv2 = AnalogConv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB1_bn2 = nn.BatchNorm2d(self.scale)
        self.LIF1_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB1_skip = AnalogConv2d(self.scale, self.scale, 1, stride=1, bias=False,rpu_config= RPU_CONFIG)

        self.SRB2_conv1 = AnalogConv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB2_bn1 = nn.BatchNorm2d(self.scale)
        self.LIF2_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_conv2 = AnalogConv2d(self.scale, self.scale, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB2_bn2 = nn.BatchNorm2d(self.scale)
        self.LIF2_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB2_skip = nn.Sequential(nn.BatchNorm2d(self.scale))    

        self.SRB3_conv1 = AnalogConv2d(self.scale, self.scale*2, 3, stride=2, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB3_bn1 = nn.BatchNorm2d(self.scale*2)
        self.LIF3_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_conv2 = AnalogConv2d(self.scale*2, self.scale*2, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB3_bn2 = nn.BatchNorm2d(self.scale*2)
        self.LIF3_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB3_skip = AnalogConv2d(self.scale, self.scale*2, 1, stride=2, bias=False,rpu_config= RPU_CONFIG) 

        self.SRB4_conv1 = AnalogConv2d(self.scale*2, self.scale*2, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB4_bn1 = nn.BatchNorm2d(self.scale*2)
        self.LIF4_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_conv2 = AnalogConv2d(self.scale*2, self.scale*2 , 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB4_bn2 = nn.BatchNorm2d(self.scale*2)
        self.LIF4_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB4_skip = nn.Sequential(nn.BatchNorm2d(self.scale*2))   

        self.SRB5_conv1 = AnalogConv2d(self.scale*2, self.scale*4, 3, stride=2, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB5_bn1 = nn.BatchNorm2d(self.scale*4)
        self.LIF5_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_conv2 = AnalogConv2d(self.scale*4, self.scale*4 , 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB5_bn2 = nn.BatchNorm2d(self.scale*4)
        self.LIF5_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB5_skip = AnalogConv2d(self.scale*2, self.scale*4, 1, stride=2, bias=False,rpu_config= RPU_CONFIG)

        self.SRB6_conv1 = AnalogConv2d(self.scale*4, self.scale*4, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB6_bn1 = nn.BatchNorm2d(self.scale*4)
        self.LIF6_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_conv2 = AnalogConv2d(self.scale*4, self.scale*4, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB6_bn2 = nn.BatchNorm2d(self.scale*4)
        self.LIF6_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB6_skip = nn.Sequential(nn.BatchNorm2d(self.scale*4))  

        self.SRB7_conv1 = AnalogConv2d(self.scale*4, self.scale*8, 3, stride=2, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB7_bn1 = nn.BatchNorm2d(self.scale*8)
        self.LIF7_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_conv2 = AnalogConv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB7_bn2 = nn.BatchNorm2d(self.scale*8)
        self.LIF7_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB7_skip = AnalogConv2d(self.scale*4, self.scale*8, 1, stride=2, bias=False,rpu_config= RPU_CONFIG)  

        self.SRB8_conv1 = AnalogConv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB8_bn1 = nn.BatchNorm2d(self.scale*8)
        self.LIF8_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_conv2 = AnalogConv2d(self.scale*8, self.scale*8, 3, stride=1, padding=1, bias=False,rpu_config= RPU_CONFIG)
        self.SRB8_bn2 = nn.BatchNorm2d(self.scale*8)
        self.LIF8_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        self.SRB8_skip = nn.Sequential(nn.BatchNorm2d(self.scale*8)) 

        # fc
        self.pool1 = nn.AvgPool2d(4)

        self.fc1 = nn.Linear(self.scale*8 , 100, bias=False)
        self.LIF_fc1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))

        self.boost1 = nn.AvgPool1d(10, 10)

        self.tau_vector = nn.Parameter(torch.ones(18, dtype=torch.float)*init_tau)
        self.tau_vector.to("cuda" if torch.cuda.is_available() else "cpu")
        self.acc_tau = torch.zeros(1) * 0.5
        
#         Attention module 1
        self.SepConv1_1 = nn.Sequential(
                nn.Conv2d(scale, scale, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale, scale, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale, affine=affine))
        self.SepConv_LIF1_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv1_2 = nn.Sequential(nn.Conv2d(scale, scale, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale, scale, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale, affine=affine))
        self.SepConv_LIF1_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))    
        nn.BatchNorm2d(64 * 1)
        self.SepConv_LIF1_3 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))  
        nn.Upsample(scale_factor=2, mode='bilinear')
        nn.Sigmoid()
#        Scala module 1
        self.SepConv2_1 = nn.Sequential(
                nn.Conv2d(scale, scale, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale, scale, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale, affine=affine))
        self.SepConv_LIF2_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv2_2 = nn.Sequential(nn.Conv2d(scale, scale, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale, scale*2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*2, affine=affine))
        self.SepConv_LIF2_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
#        Scala module 2        
        self.SepConv3_1 = nn.Sequential(
                nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*2, affine=affine))
        self.SepConv_LIF3_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv3_2 = nn.Sequential(nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF3_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True)) 
#        Scala module 3        
        self.SepConv4_1 = nn.Sequential(
                nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF4_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv4_2 = nn.Sequential(nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*8, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*8, affine=affine))
        self.SepConv_LIF4_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True)) 
        nn.AvgPool2d(4, 4)

# Attention module 2

        self.SepConv5_1 = nn.Sequential(
                nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*2, affine=affine))
        self.SepConv_LIF5_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv5_2 = nn.Sequential(nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*2, affine=affine))
        self.SepConv_LIF5_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))    
        nn.BatchNorm2d(128 * 1)
        self.SepConv_LIF5_3 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))  
        nn.Upsample(scale_factor=2, mode='bilinear')
        nn.Sigmoid()
#        Scala module 1
        self.SepConv6_1 = nn.Sequential(
                nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*2, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*2, affine=affine))
        self.SepConv_LIF6_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv6_2 = nn.Sequential(nn.Conv2d(scale*2, scale*2, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*2, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF6_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
#        Scala module 2        
        self.SepConv7_1 = nn.Sequential(
                nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF7_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv7_2 = nn.Sequential(nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4*2, affine=affine))
        self.SepConv_LIF7_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True)) 

        nn.AvgPool2d(4, 4)
# Attention module 3

        self.SepConv8_1 = nn.Sequential(
                nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF8_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv8_2 = nn.Sequential(nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*8, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF8_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))    
        nn.BatchNorm2d(256*1)
        self.SepConv_LIF8_3 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))  
        nn.Upsample(scale_factor=2, mode='bilinear')
        nn.Sigmoid()
#        Scala module 1
        self.SepConv9_1 = nn.Sequential(
                nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=stride, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*4, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*4, affine=affine))
        self.SepConv_LIF9_1 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))
        
        self.SepConv9_2 = nn.Sequential(nn.Conv2d(scale*4, scale*4, kernel_size=kernel_size, stride=1, padding=padding, groups=scale, bias=False),
                nn.Conv2d(scale*4, scale*8, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(scale*8, affine=affine))
        self.SepConv_LIF9_2 = LIF_Node(surrogate_function=HeavisideBoxcarCall(thresh=1.0, subthresh=self.subthresh, alpha=1.0, spiking=True))


        nn.AvgPool2d(4, 4)
                
#         self.attention = nn.Sequential(
#             SepConv(channel_in=64 * block.expansion, channel_out=64 * block.expansion),
#             nn.BatchNorm2d(64 * block.expansion),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid()
#         )
        
#         self.scala = nn.Sequential(
#             SepConv(channel_in=64 * block.expansion, channel_out=128 * block.expansion),
#             SepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion),
#             SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
#             nn.AvgPool2d(4, 4)
#         )
        
#         self.fc = nn.Linear(512 * block.expansion, num_classes)  # 출력 차원을 num_classes로 설정

        
        
        
    def forward(self, x):
        self.device = x.device
        # spike_recording = []
        batch_size = x.size(0)
        x = x.view(batch_size, 3, 32, 32)

        c1_mem = c1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_1_mem = srb1_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb1_2_mem = srb1_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb2_1_mem = srb2_1_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb2_2_mem = srb2_2_spike = torch.zeros(batch_size, self.scale, 32, 32, device=self.device)
        srb3_1_mem = srb3_1_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb3_2_mem = srb3_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb4_1_mem = srb4_1_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        srb4_2_mem = srb4_2_spike = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device) 
        srb5_1_mem = srb5_1_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        srb5_2_mem = srb5_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb6_1_mem = srb6_1_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        srb6_2_mem = srb6_2_spike = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device) 
        srb7_1_mem = srb7_1_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)
        srb7_2_mem = srb7_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device) 
        srb8_1_mem = srb8_1_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)
        srb8_2_mem = srb8_2_spike = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)        
        h1_mem = h1_spike = torch.zeros(batch_size, 100, device=self.device) 
        boost1 = torch.zeros(batch_size, 10, device=self.device)
        acc_mem = torch.zeros(batch_size, 10, device=self.device)

        c1_mem.fill_(0.5)
        srb1_1_mem.fill_(0.5)
        srb1_2_mem.fill_(0.5)
        srb2_1_mem.fill_(0.5)
        srb2_2_mem.fill_(0.5)
        srb3_1_mem.fill_(0.5)
        srb3_2_mem.fill_(0.5)
        srb4_1_mem.fill_(0.5)
        srb4_2_mem.fill_(0.5)
        srb5_1_mem.fill_(0.5)
        srb5_2_mem.fill_(0.5)
        srb6_1_mem.fill_(0.5)
        srb6_2_mem.fill_(0.5)
        srb7_1_mem.fill_(0.5)
        srb7_2_mem.fill_(0.5)
        srb8_1_mem.fill_(0.5)
        srb8_2_mem.fill_(0.5)
        h1_mem.fill_(0.5)

        # decay_vector = torch.sigmoid(self.tau_vector)
        # acc_decay = torch.sigmoid(self.acc_tau)
        # decay_vector = self.tau_vector
        # acc_decay = self.acc_tau

# Attetention module1_spike

         # Initialize membrane potentials and spikes for Attention module 1
        sepconv1_1_mem_1 = sepconv1_1_spike_1 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv1_1_mem_2 = sepconv1_1_spike_2 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv1_2_mem_1 = sepconv1_2_spike_1 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv1_2_mem_2 = sepconv1_2_spike_2 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 1
        sepconv2_1_mem_1 = sepconv2_1_spike_1 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv2_1_mem_2 = sepconv2_1_spike_2 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv2_2_mem_1 = sepconv2_2_spike_1 = torch.zeros(batch_size, self.scale, 16, 16, device=self.device)
        sepconv2_2_mem_2 = sepconv2_2_spike_2 = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 2
        sepconv3_1_mem_1 = sepconv3_1_spike_1 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv3_1_mem_2 = sepconv3_1_spike_2 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv3_2_mem_1 = sepconv3_2_spike_1 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv3_2_mem_2 = sepconv3_2_spike_2 = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 3
        sepconv4_1_mem_1 = sepconv4_1_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv4_1_mem_2 = sepconv4_1_spike_2 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv4_2_mem_1 = sepconv4_2_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv4_2_mem_2 = sepconv4_2_spike_2 = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)

        attention1_h1_mem = attention_h1_spike = torch.zeros(batch_size, 100, device=self.device) 
        attention1_boost = torch.zeros(batch_size, 10, device=self.device)
        attention1_acc_mem = torch.zeros(batch_size, 10, device=self.device)

        sepconv1_1_mem_1.fill_(0.5)
        sepconv1_1_mem_2.fill_(0.5)
        sepconv1_2_mem_1.fill_(0.5)
        sepconv1_2_mem_2.fill_(0.5)
        sepconv2_1_mem_1.fill_(0.5)
        sepconv2_1_mem_2.fill_(0.5)
        sepconv2_2_mem_1.fill_(0.5)
        sepconv2_2_mem_2.fill_(0.5)
        sepconv3_1_mem_1.fill_(0.5)
        sepconv3_1_mem_2.fill_(0.5)
        sepconv3_2_mem_1.fill_(0.5)
        sepconv3_2_mem_2.fill_(0.5)
        sepconv4_1_mem_1.fill_(0.5)
        sepconv4_1_mem_2.fill_(0.5)
        sepconv4_2_mem_1.fill_(0.5)
        sepconv4_2_mem_2.fill_(0.5)
        attention1_h1_mem.fill_(0.5)


# Attention module2_spike
         # Initialize membrane potentials and spikes for Attention module 2
        sepconv5_1_mem_1 = sepconv5_1_spike_1 = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        sepconv5_1_mem_2 = sepconv5_1_spike_2 = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        sepconv5_2_mem_1 = sepconv5_2_spike_1 = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)
        sepconv5_2_mem_2 = sepconv5_2_spike_2 = torch.zeros(batch_size, self.scale*2, 16, 16, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 1
        sepconv6_1_mem_1 = sepconv6_1_spike_1 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv6_1_mem_2 = sepconv6_1_spike_2 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv6_2_mem_1 = sepconv6_2_spike_1 = torch.zeros(batch_size, self.scale*2, 8, 8, device=self.device)
        sepconv6_2_mem_2 = sepconv6_2_spike_2 = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 2
        sepconv7_1_mem_1 = sepconv7_1_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv7_1_mem_2 = sepconv7_1_spike_2 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv7_2_mem_1 = sepconv7_2_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv7_2_mem_2 = sepconv7_2_spike_2 = torch.zeros(batch_size, self.scale*4*2, 4, 4, device=self.device)
        attention2_h1_mem = attention2_h1_spike = torch.zeros(batch_size, 100, device=self.device)
        attention2_boost = torch.zeros(batch_size, 10, device=self.device)
        attention2_acc_mem = torch.zeros(batch_size, 10, device=self.device)

        sepconv5_1_mem_1.fill_(0.5)
        sepconv5_1_mem_2.fill_(0.5)
        sepconv5_2_mem_1.fill_(0.5)
        sepconv5_2_mem_2.fill_(0.5)
        sepconv6_1_mem_1.fill_(0.5)
        sepconv6_1_mem_2.fill_(0.5)
        sepconv6_2_mem_1.fill_(0.5)
        sepconv6_2_mem_2.fill_(0.5)
        sepconv7_1_mem_1.fill_(0.5)
        sepconv7_1_mem_2.fill_(0.5)
        sepconv7_2_mem_1.fill_(0.5)
        sepconv7_2_mem_2.fill_(0.5)
        attention2_h1_mem.fill_(0.5)

# Attention module3_spike
        # Initialize membrane potentials and spikes for Attention module 3
        sepconv8_1_mem_1 = sepconv8_1_spike_1 = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        sepconv8_1_mem_2 = sepconv8_1_spike_2 = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        sepconv8_2_mem_1 = sepconv8_2_spike_1 = torch.zeros(batch_size, self.scale*4, 8, 8, device=self.device)
        sepconv8_2_mem_2 = sepconv8_2_spike_2 = torch.zeros(batch_size, self.scale*8, 8, 8, device=self.device)

        # Initialize membrane potentials and spikes for Scala module 1
        sepconv9_1_mem_1 = sepconv9_1_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv9_1_mem_2 = sepconv9_1_spike_2 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv9_2_mem_1 = sepconv9_2_spike_1 = torch.zeros(batch_size, self.scale*4, 4, 4, device=self.device)
        sepconv9_2_mem_2 = sepconv9_2_spike_2 = torch.zeros(batch_size, self.scale*8, 4, 4, device=self.device)

        attention3_h1_mem = attention3_h1_spike = torch.zeros(batch_size, 100, device=self.device)
        attention3_boost = torch.zeros(batch_size, 10, device=self.device)
        attention3_acc_mem = torch.zeros(batch_size, 10, device=self.device)

        sepconv8_1_mem_1.fill_(0.5)
        sepconv8_1_mem_2.fill_(0.5)
        sepconv8_2_mem_1.fill_(0.5)
        sepconv8_2_mem_2.fill_(0.5)
        sepconv9_1_mem_1.fill_(0.5)
        sepconv9_1_mem_2.fill_(0.5)
        sepconv9_2_mem_1.fill_(0.5)
        sepconv9_2_mem_2.fill_(0.5)
        attention3_h1_mem.fill_(0.5)

        decay_vector = torch.sigmoid(self.tau_vector)
        
        
        for step in range(self.num_steps):

            c1_mem, c1_spike = self.LIF1(c1_mem, c1_spike, decay_vector[0], self.conv1(x))

            srb1_1_mem, srb1_1_spike = self.LIF1_1(srb1_1_mem, srb1_1_spike, decay_vector[1], self.SRB1_bn1(self.SRB1_conv1(c1_spike)))

            srb1_2_mem, srb1_2_spike = self.LIF1_2(srb1_2_mem, srb1_2_spike, decay_vector[2], self.SRB1_bn2(self.SRB1_conv2(srb1_1_spike)+self.SRB1_skip(c1_spike)))

            srb2_1_mem, srb2_1_spike = self.LIF2_1(srb2_1_mem, srb2_1_spike, decay_vector[3], self.SRB2_bn1(self.SRB2_conv1(srb1_2_spike)))

            srb2_2_mem, srb2_2_spike = self.LIF2_2(srb2_2_mem, srb2_2_spike, decay_vector[4], self.SRB2_bn2(self.SRB2_conv2(srb2_1_spike)+self.SRB2_skip(srb1_2_spike)))
            
            # Attention module 1
            sepconv1_1_mem_1, sepconv1_1_spike_1 = self.SepConv_LIF1_1(sepconv1_1_mem_1, sepconv1_1_spike_1, decay_vector[5], self.SepConv1_1(srb2_2_spike))
            sepconv1_1_mem_2, sepconv1_1_spike_2 = self.SepConv_LIF1_2(sepconv1_1_mem_2, sepconv1_1_spike_2, decay_vector[6], self.SepConv1_2(sepconv1_1_spike_1))
            sepconv1_2_mem_1, sepconv1_2_spike_1 = self.SepConv_LIF1_1(sepconv1_2_mem_1, sepconv1_2_spike_1, decay_vector[7], self.SepConv1_1(sepconv1_1_spike_2))
            sepconv1_2_mem_2, sepconv1_2_spike_2 = self.SepConv_LIF1_2(sepconv1_2_mem_2, sepconv1_2_spike_2, decay_vector[8], self.SepConv1_2(sepconv1_2_spike_1))

            # Scala module 1
            sepconv2_1_mem_1, sepconv2_1_spike_1 = self.SepConv_LIF2_1(sepconv2_1_mem_1, sepconv2_1_spike_1, decay_vector[9], self.SepConv2_1(sepconv1_2_spike_2))
            sepconv2_1_mem_2, sepconv2_1_spike_2 = self.SepConv_LIF2_2(sepconv2_1_mem_2, sepconv2_1_spike_2, decay_vector[10], self.SepConv2_2(sepconv2_1_spike_1))
            sepconv2_2_mem_1, sepconv2_2_spike_1 = self.SepConv_LIF2_1(sepconv2_2_mem_1, sepconv2_2_spike_1, decay_vector[11], self.SepConv2_1(sepconv2_1_spike_2))
            sepconv2_2_mem_2, sepconv2_2_spike_2 = self.SepConv_LIF2_2(sepconv2_2_mem_2, sepconv2_2_spike_2, decay_vector[12], self.SepConv2_2(sepconv2_2_spike_1))

            # Scala module 2
            sepconv3_1_mem_1, sepconv3_1_spike_1 = self.SepConv_LIF3_1(sepconv3_1_mem_1, sepconv3_1_spike_1, decay_vector[13], self.SepConv3_1(sepconv2_2_spike_2))
            sepconv3_1_mem_2, sepconv3_1_spike_2 = self.SepConv_LIF3_2(sepconv3_1_mem_2, sepconv3_1_spike_2, decay_vector[14], self.SepConv3_2(sepconv3_1_spike_1))
            sepconv3_2_mem_1, sepconv3_2_spike_1 = self.SepConv_LIF3_1(sepconv3_2_mem_1, sepconv3_2_spike_1, decay_vector[15], self.SepConv3_1(sepconv3_1_spike_2))
            sepconv3_2_mem_2, sepconv3_2_spike_2 = self.SepConv_LIF3_2(sepconv3_2_mem_2, sepconv3_2_spike_2, decay_vector[16], self.SepConv3_2(sepconv3_2_spike_1))

            # Scala module 3
            sepconv4_1_mem_1, sepconv4_1_spike_1 = self.SepConv_LIF4_1(sepconv4_1_mem_1, sepconv4_1_spike_1, decay_vector[17], self.SepConv4_1(sepconv3_2_spike_2))
            sepconv4_1_mem_2, sepconv4_1_spike_2 = self.SepConv_LIF4_2(sepconv4_1_mem_2, sepconv4_1_spike_2, decay_vector[18], self.SepConv4_2(sepconv4_1_spike_1))
            sepconv4_2_mem_1, sepconv4_2_spike_1 = self.SepConv_LIF4_1(sepconv4_2_mem_1, sepconv4_2_spike_1, decay_vector[19], self.SepConv4_1(sepconv4_1_spike_2))
            sepconv4_2_mem_2, sepconv4_2_spike_2 = self.SepConv_LIF4_2(sepconv4_2_mem_2, sepconv4_2_spike_2, decay_vector[20], self.SepConv4_2(sepconv4_2_spike_1))

            # Update attention1_acc_mem
            attention1_h1_mem, attention1_h1_spike = self.LIF_fc1(attention1_h1_mem, attention1_h1_spike, decay_vector[21], self.fc1(self.pool1(sepconv4_2_spike_2).view(batch_size, -1)))
            attention1_boost = self.attention1_boost(attention1_h1_spike.unsqueeze(1)).squeeze(1)
            attention1_acc_mem += attention1_boost


            srb3_1_mem, srb3_1_spike = self.LIF3_1(srb3_1_mem, srb3_1_spike, decay_vector[22], self.SRB3_bn1(self.SRB3_conv1(srb2_2_spike)))

            srb3_2_mem, srb3_2_spike = self.LIF3_2(srb3_2_mem, srb3_2_spike, decay_vector[23], self.SRB3_bn2(self.SRB3_conv2(srb3_1_spike)+self.SRB3_skip(srb2_2_spike)))
        
            srb4_1_mem, srb4_1_spike = self.LIF4_1(srb4_1_mem, srb4_1_spike, decay_vector[24], self.SRB4_bn1(self.SRB4_conv1(srb3_2_spike)))

            srb4_2_mem, srb4_2_spike = self.LIF4_2(srb4_2_mem, srb4_2_spike, decay_vector[25], self.SRB4_bn2(self.SRB4_conv2(srb4_1_spike)+self.SRB4_skip(srb3_2_spike)))
            
            # Attention module 2
            sepconv5_1_mem_1, sepconv5_1_spike_1 = self.SepConv_LIF5_1(sepconv5_1_mem_1, sepconv5_1_spike_1, decay_vector[26], self.SepConv5_1(srb4_2_spike))
            sepconv5_1_mem_2, sepconv5_1_spike_2 = self.SepConv_LIF5_2(sepconv5_1_mem_2, sepconv5_1_spike_2, decay_vector[27], self.SepConv5_2(sepconv5_1_spike_1))
            sepconv5_2_mem_1, sepconv5_2_spike_1 = self.SepConv_LIF5_1(sepconv5_2_mem_1, sepconv5_2_spike_1, decay_vector[28], self.SepConv5_1(sepconv5_1_spike_2))
            sepconv5_2_mem_2, sepconv5_2_spike_2 = self.SepConv_LIF5_2(sepconv5_2_mem_2, sepconv5_2_spike_2, decay_vector[29], self.SepConv5_2(sepconv5_2_spike_1))

            # Scala module 1 for Attention module 2
            sepconv6_1_mem_1, sepconv6_1_spike_1 = self.SepConv_LIF6_1(sepconv6_1_mem_1, sepconv6_1_spike_1, decay_vector[30], self.SepConv6_1(sepconv5_2_spike_2))
            sepconv6_1_mem_2, sepconv6_1_spike_2 = self.SepConv_LIF6_2(sepconv6_1_mem_2, sepconv6_1_spike_2, decay_vector[31], self.SepConv6_2(sepconv6_1_spike_1))
            sepconv6_2_mem_1, sepconv6_2_spike_1 = self.SepConv_LIF6_1(sepconv6_2_mem_1, sepconv6_2_spike_1, decay_vector[32], self.SepConv6_1(sepconv6_1_spike_2))
            sepconv6_2_mem_2, sepconv6_2_spike_2 = self.SepConv_LIF6_2(sepconv6_2_mem_2, sepconv6_2_spike_2, decay_vector[33], self.SepConv6_2(sepconv6_2_spike_1))

            # Scala module 2 for Attention module 2
            sepconv7_1_mem_1, sepconv7_1_spike_1 = self.SepConv_LIF7_1(sepconv7_1_mem_1, sepconv7_1_spike_1, decay_vector[34], self.SepConv7_1(sepconv6_2_spike_2))
            sepconv7_1_mem_2, sepconv7_1_spike_2 = self.SepConv_LIF7_2(sepconv7_1_mem_2, sepconv7_1_spike_2, decay_vector[35], self.SepConv7_2(sepconv7_1_spike_1))
            sepconv7_2_mem_1, sepconv7_2_spike_1 = self.SepConv_LIF7_1(sepconv7_2_mem_1, sepconv7_2_spike_1, decay_vector[36], self.SepConv7_1(sepconv7_1_spike_2))
            sepconv7_2_mem_2, sepconv7_2_spike_2 = self.SepConv_LIF7_2(sepconv7_2_mem_2, sepconv7_2_spike_2, decay_vector[37], self.SepConv7_2(sepconv7_2_spike_1))

            # Update attention2_acc_mem
            attention2_h1_mem, attention2_h1_spike = self.LIF_fc1(attention2_h1_mem, attention2_h1_spike, decay_vector[38], self.fc1(self.pool1(sepconv7_2_spike_2).view(batch_size, -1)))
            attention2_boost = self.attention2_boost(attention2_h1_spike.unsqueeze(1)).squeeze(1)
            attention2_acc_mem += attention2_boost

            srb5_1_mem, srb5_1_spike = self.LIF5_1(srb5_1_mem, srb5_1_spike, decay_vector[39], self.SRB5_bn1(self.SRB5_conv1(srb4_2_spike)))

            srb5_2_mem, srb5_2_spike = self.LIF5_2(srb5_2_mem, srb5_2_spike, decay_vector[40], self.SRB5_bn2(self.SRB5_conv2(srb5_1_spike)+self.SRB5_skip(srb4_2_spike)))
            
            srb6_1_mem, srb6_1_spike = self.LIF6_1(srb6_1_mem, srb6_1_spike, decay_vector[41], self.SRB6_bn1(self.SRB6_conv1(srb5_2_spike)))

            srb6_2_mem, srb6_2_spike = self.LIF6_1(srb6_2_mem, srb6_2_spike, decay_vector[42], self.SRB6_bn2(self.SRB6_conv2(srb6_1_spike)+self.SRB6_skip(srb5_2_spike)))
            
            # Attention module 3
            sepconv8_1_mem_1, sepconv8_1_spike_1 = self.SepConv_LIF8_1(sepconv8_1_mem_1, sepconv8_1_spike_1, decay_vector[43], self.SepConv8_1(srb6_2_spike))
            sepconv8_1_mem_2, sepconv8_1_spike_2 = self.SepConv_LIF8_2(sepconv8_1_mem_2, sepconv8_1_spike_2, decay_vector[44], self.SepConv8_2(sepconv8_1_spike_1))
            sepconv8_2_mem_1, sepconv8_2_spike_1 = self.SepConv_LIF8_1(sepconv8_2_mem_1, sepconv8_2_spike_1, decay_vector[45], self.SepConv8_1(sepconv8_1_spike_2))
            sepconv8_2_mem_2, sepconv8_2_spike_2 = self.SepConv_LIF8_2(sepconv8_2_mem_2, sepconv8_2_spike_2, decay_vector[46], self.SepConv8_2(sepconv8_2_spike_1))

            # Scala module 1 for Attention module 3
            sepconv9_1_mem_1, sepconv9_1_spike_1 = self.SepConv_LIF9_1(sepconv9_1_mem_1, sepconv9_1_spike_1, decay_vector[47], self.SepConv9_1(sepconv8_2_spike_2))
            sepconv9_1_mem_2, sepconv9_1_spike_2 = self.SepConv_LIF9_2(sepconv9_1_mem_2, sepconv9_1_spike_2, decay_vector[48], self.SepConv9_2(sepconv9_1_spike_1))
            sepconv9_2_mem_1, sepconv9_2_spike_1 = self.SepConv_LIF9_1(sepconv9_2_mem_1, sepconv9_2_spike_1, decay_vector[49], self.SepConv9_1(sepconv9_1_spike_2))
            sepconv9_2_mem_2, sepconv9_2_spike_2 = self.SepConv_LIF9_2(sepconv9_2_mem_2, sepconv9_2_spike_2, decay_vector[50], self.SepConv9_2(sepconv9_2_spike_1))

            # Update attention3_acc_mem
            attention3_h1_mem, attention3_h1_spike = self.LIF_fc1(attention3_h1_mem, attention3_h1_spike, decay_vector[51], self.fc1(self.pool1(sepconv9_2_spike_2).view(batch_size, -1)))
            attention3_boost = self.attention3_boost(attention3_h1_spike.unsqueeze(1)).squeeze(1)
            attention3_acc_mem += attention3_boost

            srb7_1_mem, srb7_1_spike = self.LIF7_1(srb7_1_mem, srb7_1_spike, decay_vector[52], self.SRB7_bn1(self.SRB7_conv1(srb6_2_spike)))

            srb7_2_mem, srb7_2_spike = self.LIF7_2(srb7_2_mem, srb7_2_spike, decay_vector[53], self.SRB7_bn2(self.SRB7_conv2(srb7_1_spike)+self.SRB7_skip(srb6_2_spike)))
            
            srb8_1_mem, srb8_1_spike = self.LIF8_1(srb8_1_mem, srb8_1_spike, decay_vector[54], self.SRB8_bn1(self.SRB8_conv1(srb7_2_spike)))

            srb8_2_mem, srb8_2_spike = self.LIF8_2(srb8_2_mem, srb8_2_spike, decay_vector[55], self.SRB8_bn2(self.SRB8_conv2(srb8_1_spike)+self.SRB8_skip(srb7_2_spike)))
            
            h1_mem, h1_spike = self.LIF_fc1(h1_mem, h1_spike, decay_vector[56], self.fc1(self.pool1(srb8_2_spike).view(batch_size, -1)))

            boost1 = self.boost1(h1_spike.unsqueeze(1)).squeeze(1)
            

            acc_mem += boost1
#             if torch.cuda.is_available():
#                 for i in range(torch.cuda.device_count()):
#                     allocated = torch.cuda.memory_allocated(i)
#                     reserved = torch.cuda.memory_reserved(i)
#                     print(f"GPU {i}: Allocated memory: {allocated} bytes, Reserved memory: {reserved} bytes")

        return acc_mem, attention1_acc_mem, attention2_acc_mem, attention3_acc_mem
        # return next - softmax and cross-entropy loss

# class IntegratedResNet(nn.Module):
#     def __init__(self, num_classes=10, rpu_config=None):
#         super(IntegratedResNet, self).__init__()
#         # Spiking_ResNet18_LIF_STBP 모델을 생성하고 아날로그 변환을 적용합니다.
#         self.spiking_resnet = Spiking_ResNet18_LIF_STBP()
        
#         rpu_config_float = FloatingPointRPUConfig()
#         self.spiking_resnet = convert_to_analog(self.spiking_resnet, rpu_config=rpu_config_float)

#     def forward(self, x):
#         x = self.spiking_resnet(x)
#         return x
# class IntegratedResNet(nn.Module):
#     def __init__(self, architecture="resnet10", num_classes=10, rpu_config=None):
#         super(IntegratedResNet, self).__init__()
#         self.backbone = create_resnet(architecture, num_classes)
#         self.backbone = convert_to_analog(self.backbone, rpu_config)

#         # 어텐션 모듈들 생성
#         block_type = 'BasicBlock' if architecture in ["resnet18", "resnet34", "resnet10"] else 'Bottleneck'
#         self.attention1 = ResNetAttention1(block_type, num_classes)
#         self.attention2 = ResNetAttention2(block_type, num_classes)
#         self.attention3 = ResNetAttention3(block_type, num_classes)

#         # rpu_config_float = FloatingPointRPUConfig()
#         # self.attention1 = convert_to_analog(self.attention1, rpu_config_float)
#         # self.attention2 = convert_to_analog(self.attention2, rpu_config_float) 
#         # self.attention3 = convert_to_analog(self.attention3, rpu_config_float)

#     def forward(self, x):
#         # ResNet 백본 통과
#         out4, feature, x1, x2, x3 = self.backbone(x)

#         # 어텐션 모듈들을 각각의 특성 맵에 적용
#         out1, feature1 = self.attention1(x1)
#         out2, feature2 = self.attention2(x2)
#         out3, feature3 = self.attention3(x3)

#         return out1, feature1, out2, feature2, out3, feature3, out4, feature
# class ResNetAttention1(nn.Module):
#     def __init__(self, block, num_classes=100):  # num_classes 매개변수 추가
#         super(ResNetAttention1, self).__init__()
#         block = self._get_block_class(block)

#         self.attention = nn.Sequential(
#             SepConv(channel_in=64 * block.expansion, channel_out=64 * block.expansion),
#             nn.BatchNorm2d(64 * block.expansion),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid()
#         )
        
#         self.scala = nn.Sequential(
#             SepConv(channel_in=64 * block.expansion, channel_out=128 * block.expansion),
#             SepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion),
#             SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
#             nn.AvgPool2d(4, 4)
#         )
        
#         self.fc = nn.Linear(512 * block.expansion, num_classes)  # 출력 차원을 num_classes로 설정

#     def forward(self, x1):
#         fea = self.attention(x1)
#         fea = fea * x1
#         feature_out = self.scala(fea).view(fea.size(0), -1)
#         out = self.fc(feature_out)
#         return out, feature_out
    
#     @staticmethod
#     def _get_block_class(block_name):
#         if block_name == 'BasicBlock':
#             return BasicBlock
#         elif block_name == 'Bottleneck':
#             return Bottleneck
#         else:
#             raise ValueError(f"Unknown block type: {block_name}")
    
# class ResNetAttention2(nn.Module):
#     def __init__(self, block, num_classes=100):
#         super(ResNetAttention2, self).__init__()
#         block = self._get_block_class(block)

#         self.attention = nn.Sequential(
#             SepConv(channel_in=128 * block.expansion, channel_out=128 * block.expansion),
#             nn.BatchNorm2d(128 * block.expansion),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid()
#         )
        
#         self.scala = nn.Sequential(
#             SepConv(channel_in=128 * block.expansion, channel_out=256 * block.expansion),
#             SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
#             nn.AvgPool2d(4, 4)
#         )
        
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def forward(self, x2):
#         fea = self.attention(x2)
#         fea = fea * x2
#         feature_out = self.scala(fea).view(fea.size(0), -1)
#         out = self.fc(feature_out)
#         return out, feature_out

#     @staticmethod
#     def _get_block_class(block_name):
#         if block_name == 'BasicBlock':
#             return BasicBlock
#         elif block_name == 'Bottleneck':
#             return Bottleneck
#         else:
#             raise ValueError(f"Unknown block type: {block_name}")
        
# class ResNetAttention3(nn.Module):
#     def __init__(self, block, num_classes=100):
#         super(ResNetAttention3, self).__init__()
#         block = self._get_block_class(block)
        
#         self.attention = nn.Sequential(
#             SepConv(channel_in=256 * block.expansion, channel_out=256 * block.expansion),
#             nn.BatchNorm2d(256 * block.expansion),
#             nn.ReLU(),
#             nn.Upsample(scale_factor=2, mode='bilinear'),
#             nn.Sigmoid()
#         )
        
#         self.scala = nn.Sequential(
#             SepConv(channel_in=256 * block.expansion, channel_out=512 * block.expansion),
#             nn.AvgPool2d(4, 4)
#         )
        
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#     def forward(self, x3):
#         fea = self.attention(x3)
#         fea = fea * x3
#         feature_out = self.scala(fea).view(fea.size(0), -1)
#         out = self.fc(feature_out)
#         return out, feature_out
    
#     @staticmethod
#     def _get_block_class(block_name):
#         if block_name == 'BasicBlock':
#             return BasicBlock
#         elif block_name == 'Bottleneck':
#             return Bottleneck
#         else:
#             raise ValueError(f"Unknown block type: {block_name}")
