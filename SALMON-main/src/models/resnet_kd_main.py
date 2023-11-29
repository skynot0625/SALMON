import torch.optim as optim
import os
from datetime import datetime
import os
import pandas as pd
from datetime import datetime
# Imports from PyTorch.
from torch import nn, Tensor, device, no_grad, manual_seed, save
from torch import max as torch_max
from torch.utils.data import DataLoader
import torch.nn.functional as F
# from analog_resnet import AnalogBasicBlock, AnalogBottleneck, AnalogResNetBackbone, create_analog_resnet
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
# Imports from aihwkit.
from analog_resnet import *
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaEcRamPreset, IdealizedPreset, EcRamPreset
from aihwkit.simulator.configs import MappingParameter
from aihwkit.simulator.rpu_base import cuda
from aihwkit.simulator.presets import GokmenVlasovPreset
from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    UnitCellRPUConfig,
    SingleRPUConfig,
    BufferedTransferCompound,
    SoftBoundsDevice,
    ConstantStepDevice,
    MappingParameter,
    IOParameters,
    UpdateParameters,IdealDevice
)
from aihwkit.simulator.configs  import SoftBoundsDevice, SoftBoundsPmaxDevice
from analog_resnet import AnalogResNetBackbone, AnalogBasicBlock,create_analog_resnet
import random
import numpy as np

# Argument class to store hyperparameters and configurations
USE_CUDA = 0
if torch.cuda.is_available():
    USE_CUDA = 1
from resnet_kd_module import *
# DEVICE 정의
DEVICE = device("cuda" if USE_CUDA else "cpu")

# Path to store datasets
PATH_DATASET = os.path.join(os.getcwd(), "data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "RESNET")
WEIGHT_PATH = os.path.join(RESULTS, "example_18_model_weight.pth")

mapping = MappingParameter(weight_scaling_omega=0.6)
# RPU_CONFIG = TikiTakaEcRamPreset(mapping=mapping)
# RPU_CONFIG = IdealizedPreset()

class Args:
    model = "resnet10"
    dataset = "cifar10"
    epoch = 300
    loss_coefficient = 0.3
    feature_loss_coefficient = 0.3
    dataset_path = "data"
    autoaugment = False
    temperature = 3.0
    batchsize = 128
    init_lr = 0.1
    N_CLASSES = 10
    block = BasicBlock

    # Add alpha and p_max to Args
    alpha =0.3
    p_max = 1000

args = Args()
print(args)

# device_config = SoftBoundsPmaxDevice(alpha=0.001 / 2, p_max = 1000)
# RPU_CONFIG = SingleRPUConfig(
#     device=device_config,
#     update=UpdateParameters(desired_bl=10),
#     mapping=mapping
# )
# RPU_CONFIG = TikiTakaEcRamPreset(mapping=mapping)

def main():
    # 고정된 시드 값으로 난수 생성기 초기화
    set_seed(42)

    # CIFAR 데이터셋 로드 및 dataloaders 생성
    trainloader, testloader = load_images(args)

    # 모델 인스턴스 생성
    net_attention1, net_attention2, net_attention3 = create_attention_resnets(architecture=args.model, num_classes=args.N_CLASSES)
    net_attention1.to(DEVICE)
    net_attention2.to(DEVICE)
    net_attention3.to(DEVICE)

    net_backbone = create_analog_resnet(architecture=args.model, num_classes=args.N_CLASSES, rpu_config=RPU_CONFIG).to(DEVICE)
    net_digital = create_resnet(architecture=args.model, num_classes=args.N_CLASSES).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()

    params_to_update = list(net_backbone.parameters()) + list(net_attention1.parameters()) + list(net_attention2.parameters()) + list(net_attention3.parameters())
    optimizer = AnalogAdam(params_to_update, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=5e-4, amsgrad=False)

    # Salmon 알고리즘 적용
    epochs = args.epoch  # 전체 훈련 에포크
    distill_epoch =1 # 10 에포크마다 distillation을 진행하도록 설정

    # apply_salmon 함수를 사용하여 주기적인 distillation과 일반 훈련을 번갈아 수행

    student_model, optimizer, results = training_loop_distillation(
        student_model=net_backbone,
        teacher_model=net_digital,
        criterion=criterion,
        optimizer=optimizer,
        train_data=trainloader,
        validation_data=testloader,
        temperature=args.temperature,
        alpha=args.alpha,
        epochs=args.epochs,
        distill_every=args.distill_every,
        print_every=args.print_every
    )    
    # 훈련이 끝난 후 메모리 정리
    del net_backbone, net_attention1, net_attention2, net_attention3, optimizer, criterion
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()



