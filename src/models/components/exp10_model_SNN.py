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
# RPU_CONFIG = SingleRPUConfig(device=IdealDevice())
RPU_CONFIG = IdealizedPreset()
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
        subthresh  = 1.0,
        RPU_CONFIG = None
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

        decay_vector = torch.sigmoid(self.tau_vector)
        # acc_decay = torch.sigmoid(self.acc_tau)
        # decay_vector = self.tau_vector
        # acc_decay = self.acc_tau

        for step in range(self.num_steps):

            c1_mem, c1_spike = self.LIF1(c1_mem, c1_spike, decay_vector[0], self.conv1(x))

            srb1_1_mem, srb1_1_spike = self.LIF1_1(srb1_1_mem, srb1_1_spike, decay_vector[1], self.SRB1_bn1(self.SRB1_conv1(c1_spike)))

            srb1_2_mem, srb1_2_spike = self.LIF1_2(srb1_2_mem, srb1_2_spike, decay_vector[2], self.SRB1_bn2(self.SRB1_conv2(srb1_1_spike)+self.SRB1_skip(c1_spike)))

            srb2_1_mem, srb2_1_spike = self.LIF2_1(srb2_1_mem, srb2_1_spike, decay_vector[3], self.SRB2_bn1(self.SRB2_conv1(srb1_2_spike)))

            srb2_2_mem, srb2_2_spike = self.LIF2_2(srb2_2_mem, srb2_2_spike, decay_vector[4], self.SRB2_bn2(self.SRB2_conv2(srb2_1_spike)+self.SRB2_skip(srb1_2_spike)))

            srb3_1_mem, srb3_1_spike = self.LIF3_1(srb3_1_mem, srb3_1_spike, decay_vector[5], self.SRB3_bn1(self.SRB3_conv1(srb2_2_spike)))

            srb3_2_mem, srb3_2_spike = self.LIF3_2(srb3_2_mem, srb3_2_spike, decay_vector[6], self.SRB3_bn2(self.SRB3_conv2(srb3_1_spike)+self.SRB3_skip(srb2_2_spike)))
        
            srb4_1_mem, srb4_1_spike = self.LIF4_1(srb4_1_mem, srb4_1_spike, decay_vector[7], self.SRB4_bn1(self.SRB4_conv1(srb3_2_spike)))

            srb4_2_mem, srb4_2_spike = self.LIF4_2(srb4_2_mem, srb4_2_spike, decay_vector[8], self.SRB4_bn2(self.SRB4_conv2(srb4_1_spike)+self.SRB4_skip(srb3_2_spike)))
            
            srb5_1_mem, srb5_1_spike = self.LIF5_1(srb5_1_mem, srb5_1_spike, decay_vector[9], self.SRB5_bn1(self.SRB5_conv1(srb4_2_spike)))

            srb5_2_mem, srb5_2_spike = self.LIF5_2(srb5_2_mem, srb5_2_spike, decay_vector[10], self.SRB5_bn2(self.SRB5_conv2(srb5_1_spike)+self.SRB5_skip(srb4_2_spike)))
            
            srb6_1_mem, srb6_1_spike = self.LIF6_1(srb6_1_mem, srb6_1_spike, decay_vector[11], self.SRB6_bn1(self.SRB6_conv1(srb5_2_spike)))

            srb6_2_mem, srb6_2_spike = self.LIF6_1(srb6_2_mem, srb6_2_spike, decay_vector[12], self.SRB6_bn2(self.SRB6_conv2(srb6_1_spike)+self.SRB6_skip(srb5_2_spike)))
            
            srb7_1_mem, srb7_1_spike = self.LIF7_1(srb7_1_mem, srb7_1_spike, decay_vector[13], self.SRB7_bn1(self.SRB7_conv1(srb6_2_spike)))

            srb7_2_mem, srb7_2_spike = self.LIF7_2(srb7_2_mem, srb7_2_spike, decay_vector[14], self.SRB7_bn2(self.SRB7_conv2(srb7_1_spike)+self.SRB7_skip(srb6_2_spike)))
            
            srb8_1_mem, srb8_1_spike = self.LIF8_1(srb8_1_mem, srb8_1_spike, decay_vector[15], self.SRB8_bn1(self.SRB8_conv1(srb7_2_spike)))

            srb8_2_mem, srb8_2_spike = self.LIF8_2(srb8_2_mem, srb8_2_spike, decay_vector[16], self.SRB8_bn2(self.SRB8_conv2(srb8_1_spike)+self.SRB8_skip(srb7_2_spike)))
            
            h1_mem, h1_spike = self.LIF_fc1(h1_mem, h1_spike, decay_vector[17], self.fc1(self.pool1(srb8_2_spike).view(batch_size, -1)))

            boost1 = self.boost1(h1_spike.unsqueeze(1)).squeeze(1)

            acc_mem += boost1
#             if torch.cuda.is_available():
#                 for i in range(torch.cuda.device_count()):
#                     allocated = torch.cuda.memory_allocated(i)
#                     reserved = torch.cuda.memory_reserved(i)
#                     print(f"GPU {i}: Allocated memory: {allocated} bytes, Reserved memory: {reserved} bytes")
        return acc_mem
        # return next - softmax and cross-entropy loss

class IntegratedResNet(nn.Module):
    def __init__(self, num_classes=10, rpu_config=None):
        super(IntegratedResNet, self).__init__()
        # Spiking_ResNet18_LIF_STBP 모델을 생성하고 아날로그 변환을 적용합니다.
        self.spiking_resnet = Spiking_ResNet18_LIF_STBP()
        
        rpu_config_float = FloatingPointRPUConfig()
        self.spiking_resnet = convert_to_analog(self.spiking_resnet, rpu_config=rpu_config_float)

    def forward(self, x):
        x = self.spiking_resnet(x)
        return x
