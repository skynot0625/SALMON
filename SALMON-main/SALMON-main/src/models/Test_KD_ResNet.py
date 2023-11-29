import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
from torch import nn, Tensor, device, no_grad, manual_seed
from torch import max as torch_max
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.optim import SGD
from aihwkit.nn import AnalogConv2d, AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
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
    UpdateParameters,
)
from aihwkit.simulator.configs  import SoftBoundsDevice, SoftBoundsPmaxDevice
from aihwkit.simulator.rpu_base import cuda
import pandas as pd
import torch.nn.functional as F

USE_CUDA = 0
if cuda.is_compiled():
    USE_CUDA = 1
DEVICE = device("cuda" if USE_CUDA else "cpu")
print("Model is on:", DEVICE)
# Path to store datasets
PATH_DATASET = os.path.join("data", "DATASET")

# Path to store results
RESULTS = os.path.join(os.getcwd(), "results", "VGG8")
# Training parameters
SEED = 1
N_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.1
N_CLASSES = 10
WEIGHT_SCALING_OMEGA = 0.6  # Should not be larger than max weight.
RESULTS_TEST = "results"
temperature =2.0
alpha = 0.1
# Select the device model to use in the training. In this case we are using one of the preset,
# but it can be changed to a number of preset to explore possible different analog devices
mapping = MappingParameter(weight_scaling_omega=WEIGHT_SCALING_OMEGA)
device_config = SoftBoundsPmaxDevice(alpha=0.001 / 2, p_max = 1000)
rpu_config = SingleRPUConfig(
    device=device_config,
    update=UpdateParameters(desired_bl=10),
    mapping=mapping
)

##--cifar 10 data set--
def load_images():
    """Load images for train from torchvision datasets."""
    mean = Tensor([0.4914, 0.4822, 0.4465])
    std = Tensor([0.2470, 0.2435, 0.2616])

    print(f"Normalization data: ({mean},{std})")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    
    # CIFAR-10 datasets
    train_set = datasets.SVHN(root=PATH_DATASET, train=True, download=True, transform=transform)
    val_set = datasets.CIFAR10(root=PATH_DATASET, train=False, download=True, transform=transform)

    train_data = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    validation_data = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_data, validation_data



def load_teacher_model(model_name, pretrained=True):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)

        raise ValueError(f"Unknown model name: {model_name}")
    return model


def save_to_excel(filename, train_losses, valid_losses, test_error, accuracies):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_filename = f"{filename}_{timestamp_str}.xlsx"
    df = pd.DataFrame({
        'Train Loss': train_losses,
        'Valid Loss': valid_losses,
        'Test Error': test_error,
        'Accuracy': accuracies
    })
    df.to_excel(full_filename, index=False)

def plot_results(train_losses, valid_losses, test_error):
    """Plot results.

    Args:
        train_losses (List): training losses as calculated in the training_loop
        valid_losses (List): validation losses as calculated in the training_loop
        test_error (List): test error as calculated in the training_loop
    """
    fig = plt.plot(train_losses, "r-s", valid_losses, "b-o")
    plt.title("aihwkit VGG8")
    plt.legend(fig[:2], ["Training Losses", "Validation Losses"])
    plt.xlabel("Epoch number")
    plt.ylabel("Loss [A.U.]")
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_losses.png"))
    plt.close()

    fig = plt.plot(test_error, "r-s")
    plt.title("aihwkit VGG8")
    plt.legend(fig[:1], ["Test Error"])
    plt.xlabel("Epoch number")
    plt.ylabel("Test Error [%]")
    plt.yscale("log")
    plt.ylim((5e-1, 1e2))
    plt.grid(which="both", linestyle="--")
    plt.savefig(os.path.join(RESULTS, "test_error.png"))
    plt.close()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def create_custom_ResNet(version=18, num_classes=1000):
    block = BasicBlock
    layers = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3]
    }[version]
    
    return _make_resnet(block, layers, num_classes)


def _make_resnet(block, layers, num_classes):
    in_planes = 64
    model_layers = [
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    ]

    # Make layers
    model_layers += _make_layer(block, 64, layers[0], stride=1)
    model_layers += _make_layer(block, 128, layers[1], stride=2)
    model_layers += _make_layer(block, 256, layers[2], stride=2)
    model_layers += _make_layer(block, 512, layers[3], stride=2)
    model_layers += [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]

    model_layers += [nn.Linear(512 * block.expansion, num_classes)]

    return nn.Sequential(*model_layers)


def _make_layer(block, planes, num_blocks, stride):
    layers = []
    strides = [stride] + [1] * (num_blocks - 1)
    in_planes = _make_layer.in_planes  # use as a static variable
    for stride in strides:
        layers.append(block(in_planes, planes, stride))
        in_planes = planes * block.expansion
    _make_layer.in_planes = in_planes  # update the static variable
    return layers

_make_layer.in_planes = 64  # initialize the static variable

def create_analog_VGG8():
    """Create a Vgg8 inspired analog model.

    Returns:
       nn.Module: VGG8 model
    """
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = AnalogSequential(
        nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[0],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        AnalogConv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.ReLU(),
        AnalogConv2d(
            in_channels=channel[2],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
            rpu_config=RPU_CONFIG,
        ),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        AnalogLinear(in_features=16 * channel[2], out_features=fc_size, rpu_config=RPU_CONFIG),
        nn.ReLU(),
        nn.Linear(in_features=fc_size, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    )
    return model

def create_analog_CNN5():
    num_channels = 32
    channel = [num_channels, num_channels * 2, num_channels * 4]
    
    model = AnalogSequential(
        # Conv Layer 1
        AnalogConv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1, rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Conv Layer 2
        AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=1, padding=1, rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Conv Layer 3
        AnalogConv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, stride=1, padding=1, rpu_config=RPU_CONFIG),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Flatten
        nn.Flatten(),
        
        # Fully Connected Layer 1
        AnalogLinear(in_features=4 * 4 * channel[2], out_features=channel[2], rpu_config=RPU_CONFIG),
        nn.BatchNorm1d(channel[2]),
        nn.ReLU(),
        
        # Fully Connected Layer 2
        nn.Linear(in_features=channel[2], out_features=N_CLASSES),
        
        # LogSoftmax
        nn.LogSoftmax(dim=1)
    )
    
    return model

def create_custom_VGG(version=8, channel_base=64):
    """
    Creates a VGG-inspired model with a custom number of layers.

    Args:
        version (int): The version number corresponds to different layer configurations.
        channel_base (int): The number of channels in the first convolutional layer. 
                            Subsequent layers may have a multiple of this number.
                      
    Returns:
        nn.Module: A VGG-like model
    """
    
    layer_config = {
        7: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 'M'],
        8: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        11: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        13: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        16: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 8 * channel_base, 'M'],
        19: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 8 * channel_base, 8 * channel_base, 'M']
    }
    
    config = layer_config.get(version, layer_config[8])
    layers = []
    in_channels = 3
    
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)]
        else:
            layers += [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=v,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(v)
            ]
            in_channels = v
    
    layers += [nn.Flatten()]
    fc_size = ((32 // (2 ** (config.count('M')))) ** 2) * in_channels  # calculate the flattened size
    layers += [
        nn.Linear(in_features=fc_size, out_features=4096),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    ]
    
    return nn.Sequential(*layers)



def create_custom_analog_VGG(version=8, channel_base=64):
    """
    Creates a VGG-inspired model with a custom number of layers.

    Args:
        version (int): The version number corresponds to different layer configurations.
        channel_base (int): The number of channels in the first convolutional layer. 
                            Subsequent layers may have a multiple of this number.
                      
    Returns:
        nn.Module: A VGG-like model
    """
    
    layer_config = {
        7: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 'M'],
        8: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        11: [channel_base, 'M', 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        13: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 'M'],
        16: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 8 * channel_base, 'M'],
        19: [channel_base, channel_base, 'M', 2 * channel_base, 2 * channel_base, 'M', 4 * channel_base, 4 * channel_base, 4 * channel_base, 4 * channel_base, 'M', 8 * channel_base, 8 * channel_base, 8 * channel_base, 8 * channel_base, 'M']
    }
    
    config = layer_config.get(version, layer_config[8])
    layers = []
    in_channels = 3
    
    for v in config:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)]
        else:
            layers += [
                AnalogConv2d(
                    in_channels=in_channels,
                    out_channels=v,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    rpu_config=RPU_CONFIG,
                ),
                nn.ReLU(),
                nn.BatchNorm2d(v)
            ]
            in_channels = v
    
    layers += [nn.Flatten()]
    fc_size = ((32 // (2 ** (config.count('M')))) ** 2) * in_channels  # calculate the flattened size
    layers += [
        AnalogLinear(in_features=fc_size, out_features=4096, rpu_config=RPU_CONFIG),
        nn.ReLU(),
        nn.Linear(in_features=4096, out_features=N_CLASSES),
        nn.LogSoftmax(dim=1),
    ]
    
    return AnalogSequential(*layers)


# 사용 예시:
def create_VGG8():
    """Create a Vgg8 inspired digital model.

    Returns:
       nn.Module: VGG8 model
    """
    channel_base = 48
    channel = [channel_base, 2 * channel_base, 3 * channel_base]
    fc_size = 8 * channel_base
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[0],
            out_channels=channel[0],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Conv2d(
            in_channels=channel[0],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[1],
            out_channels=channel[1],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Conv2d(
            in_channels=channel[1],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=channel[2],
            out_channels=channel[2],
            kernel_size=3,
            stride=1,
            padding=1,
        ),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
        nn.Flatten(),
        nn.Linear(in_features=16 * channel[2], out_features=fc_size),
        nn.ReLU(),
        nn.Linear(in_features=fc_size, out_features=10), # 예를 들어 10개의 클래스가 있다고 가정했습니다.
        nn.LogSoftmax(dim=1),
    )
    return model

def create_CNN5():
    num_channels = 32
    channel = [num_channels, num_channels * 2, num_channels * 4]
    
    model = nn.Sequential(
        # Conv Layer 1
        nn.Conv2d(in_channels=3, out_channels=channel[0], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[0]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Conv Layer 2
        nn.Conv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[1]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Conv Layer 3
        nn.Conv2d(in_channels=channel[1], out_channels=channel[2], kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(channel[2]),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        
        # Flatten
        nn.Flatten(),
        
        # Fully Connected Layer 1
        nn.Linear(in_features=4 * 4 * channel[2], out_features=channel[2]),
        nn.BatchNorm1d(channel[2]),
        nn.ReLU(),
        
        # Fully Connected Layer 2
        nn.Linear(in_features=channel[2], out_features=N_CLASSES),
        
        # LogSoftmax
        nn.LogSoftmax(dim=1)
    )
    
    return model


def create_analog_sgd_optimizer(model, learning_rate):

    optimizer = AnalogSGD(model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(model)

    return optimizer

def create_sgd_optimizer(model, learning_rate):

    optimizer = SGD(model.parameters(), lr=learning_rate)
    return optimizer

def train_step(train_data, model, criterion, optimizer):

    total_loss = 0

    model.train()

    for images, labels in train_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()

        # Add training Tensor to the model (input).
        output = model(images)
        loss = criterion(output, labels)

        # Run training (backward propagation).
        loss.backward()

        # Optimize weights.
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    epoch_loss = total_loss / len(train_data.dataset)

    return model, optimizer, epoch_loss

def test_evaluation(validation_data, model, criterion):

    total_loss = 0
    predicted_ok = 0
    total_images = 0

    model.eval()

    for images, labels in validation_data:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        pred = model(images)
        loss = criterion(pred, labels)
        total_loss += loss.item() * images.size(0)


        _, predicted = torch_max(pred.data, 1)
        total_images += labels.size(0)
        predicted_ok += (predicted == labels).sum().item()
        accuracy = predicted_ok / total_images * 100
        error = (1 - predicted_ok / total_images) * 100

    epoch_loss = total_loss / len(validation_data.dataset)

    return model, epoch_loss, error, accuracy

def distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha):

    soft_loss = F.kl_div(
        F.log_softmax(student_outputs / temperature, dim=1),
        F.softmax(teacher_outputs / temperature, dim=1),
        reduction='batchmean',
    ) * (temperature * temperature)
    
    hard_loss = F.cross_entropy(student_outputs, labels)
    
    return alpha * hard_loss + (1. - alpha) * soft_loss

def train_distillation(student_model, teacher_model, train_loader, optimizer, temperature, alpha):
    student_model.train()
    teacher_model.eval()
    
    total_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        # images = images.view(images.shape[0], -1)  # 이 줄을 제거하세요
        
        optimizer.zero_grad()
        student_outputs = student_model(images)
        with torch.no_grad():
            teacher_outputs = teacher_model(images)
        
        loss = distillation_loss(student_outputs, teacher_outputs, labels, temperature, alpha)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    
    return average_loss  # Return the average loss for this epoch


def training_loop(model, criterion, optimizer, train_data, validation_data, epochs, print_every=1):
    train_losses = []
    valid_losses = []
    test_error = []
    accuracies = []  # To store accuracies for each epoch

    # Train model
    for epoch in range(0, epochs):
        # Train_step
        model, optimizer, train_loss = train_step(train_data, model, criterion, optimizer)
        train_losses.append(train_loss)

        if epoch % print_every == (print_every - 1):
            # Validate_step
            with torch.no_grad():
                model, valid_loss, error, accuracy = test_evaluation(
                    validation_data, model, criterion
                )
                valid_losses.append(valid_loss)
                test_error.append(error)
                accuracies.append(accuracy)  # Store the accuracy for this epoch

            print(
                f"{datetime.now().time().replace(microsecond=0)} --- "
                f"Epoch: {epoch}\t"
                f"Train loss: {train_loss:.4f}\t"
                f"Valid loss: {valid_loss:.4f}\t"
                f"Test error: {error:.2f}%\t"
                f"Test accuracy: {accuracy:.2f}%\t"
            )
    plot_results(train_losses, valid_losses, test_error)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    test_error_file = os.path.join(RESULTS, f"Test_error_{timestamp_str}.csv")
    train_losses_file = os.path.join(RESULTS, f"Train_Losses_{timestamp_str}.csv")
    valid_losses_file = os.path.join(RESULTS, f"Valid_Losses_{timestamp_str}.csv")

    np.savetxt(test_error_file, test_error, delimiter=",")
    np.savetxt(train_losses_file, train_losses, delimiter=",")
    np.savetxt(valid_losses_file, valid_losses, delimiter=",")
    return model, optimizer, (train_losses, valid_losses, test_error, accuracies)  # Return accuracies as well

def training_loop_distillation(student_model, teacher_model, criterion, optimizer, train_data, validation_data, temperature, alpha, epochs, distill_every=1, print_every=1):
    train_losses = []
    valid_losses = []
    test_error = []
    accuracies = []  # To store accuracies for each epoch
    
    # Train model
    for epoch in range(epochs):
        if epoch % distill_every == 0:  # If the current epoch is a multiple of distill_every, perform distillation
            average_loss = train_distillation(student_model, teacher_model, train_data, optimizer, temperature, alpha)
        else:  # Otherwise, perform standard training
            student_model, optimizer, average_loss = train_step(train_data, student_model, criterion, optimizer)
            
        train_losses.append(average_loss)
        
        # Validation
        if epoch % print_every == (print_every - 1):
            with torch.no_grad():
                student_model.eval()  # Set student_model to evaluation mode
                _, valid_loss, error, accuracy = test_evaluation(validation_data, student_model, criterion)
                valid_losses.append(valid_loss)
                test_error.append(error)
                accuracies.append(accuracy)  # Store the accuracy for this epoch

            print(f"{datetime.now().time().replace(microsecond=0)} --- "
                  f"Epoch: {epoch}\t"
                  f"Train loss: {average_loss:.4f}\t"
                  f"Valid loss: {valid_loss:.4f}\t"
                  f"Test error: {error:.2f}%\t"
                  f"Test accuracy: {accuracy:.2f}%\t")
    plot_results(train_losses, valid_losses, test_error)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    test_error_file = os.path.join(RESULTS, f"Test_error_{timestamp_str}.csv")
    train_losses_file = os.path.join(RESULTS, f"Train_Losses_{timestamp_str}.csv")
    valid_losses_file = os.path.join(RESULTS, f"Valid_Losses_{timestamp_str}.csv")

    np.savetxt(test_error_file, test_error, delimiter=",")
    np.savetxt(train_losses_file, train_losses, delimiter=",")
    np.savetxt(valid_losses_file, valid_losses, delimiter=",")
    return student_model, optimizer, (train_losses, valid_losses, test_error, accuracies)  # Return accuracies as well



def main():
    os.makedirs(RESULTS, exist_ok=True)
    manual_seed(SEED)
    train_data, validation_data = load_images()

    teacher_model = create_custom_VGG(version=16)
    teacher_model = teacher_model.to(DEVICE)
    optimizer_teacher = create_sgd_optimizer(teacher_model, LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{datetime.now().time().replace(microsecond=0)} --- " f"Started VGG8 Example")
    print("Training Teacher Model...")
    teacher_model, optimizer_teacher, (train_losses_t, valid_losses_t, test_error_t, accuracies_t) = training_loop(
        teacher_model, criterion, optimizer_teacher, train_data, validation_data, N_EPOCHS
    )
    save_to_excel(os.path.join(RESULTS_TEST, 'teacher_model_results.xlsx'),
                  train_losses_t, valid_losses_t, test_error_t, accuracies_t)

    # distill_every_values = [1, 2, 5, 10, 20]
    # for distill_value in distill_every_values:
    #     student_model_kd = create_custom_analog_VGG(8).to(DEVICE)
    #     optimizer_student_kd = create_analog_sgd_optimizer(student_model_kd, LEARNING_RATE)
    #     print(f"Training Student Model with KD and distill_every={distill_value}...")
    #     student_model_kd, optimizer_student_kd, (train_losses_kd, valid_losses_kd, test_error_kd, accuracies_kd) = training_loop_distillation(
    #         student_model_kd, teacher_model, criterion, optimizer_student_kd, train_data, validation_data, temperature, alpha, N_EPOCHS, distill_every=distill_value
    #     )
    #     save_to_excel(os.path.join(RESULTS_TEST, f'student_model_kd_results_distill_{distill_value}.xlsx'),
    #                   train_losses_kd, valid_losses_kd, test_error_kd, accuracies_kd)

    # ... [The rest of your code remains unchanged]
    # student_model = create_custom_analog_VGG(8).to(DEVICE)
    # optimizer_student = create_analog_sgd_optimizer(student_model, LEARNING_RATE)
    # print("Training Analog Student Model without KD...")
    # student_model, optimizer_student, (train_losses_s, valid_losses_s, test_error_s, accuracies_s) = training_loop(
    #     student_model, criterion, optimizer_student, train_data, validation_data, N_EPOCHS
    # )
    # save_to_excel(os.path.join(RESULTS_TEST, 'student_model_results.xlsx'), train_losses_s, valid_losses_s, test_error_s, accuracies_s)

    student_model_digit = create_custom_VGG(8).to(DEVICE)
    optimizer_student_digit = create_sgd_optimizer(student_model_digit, LEARNING_RATE)
    print("Training Digital Student Model...")
    student_model_digit, optimizer_student_digit, (train_losses_d, valid_losses_d, test_error_d, accuracies_d) = training_loop(
            student_model_digit, criterion, optimizer_student_digit, train_data, validation_data, N_EPOCHS
        )
    save_to_excel(os.path.join(RESULTS_TEST, 'student_model_digit_results.xlsx'), train_losses_d, valid_losses_d, test_error_d, accuracies_d)
    print("Training Digital Student Model with KD...")
    distill_every_values = [1, 2, 5, 10, 20]
    for distill_value in distill_every_values:
        student_model_digit = create_custom_VGG(8).to(DEVICE)
        optimizer_student = create_sgd_optimizer(student_model_digit, LEARNING_RATE)
        print(f"Training Student Model with KD and distill_every={distill_value}...")
        student_model_digit, optimizer_student, (train_losses_kd, valid_losses_kd, test_error_kd, accuracies_kd) = training_loop_distillation(
            student_model_digit, teacher_model, criterion, optimizer_student, train_data, validation_data, temperature, alpha, N_EPOCHS, distill_every=distill_value
        )
        save_to_excel(os.path.join(RESULTS_TEST, f'student_model_kd_results_distill_{distill_value}.xlsx'),
                      train_losses_kd, valid_losses_kd, test_error_kd, accuracies_kd)

    print(f"{datetime.now().time().replace(microsecond=0)} --- " f"Completed VGG8 Example")

    
if __name__ == "__main__":
    main()
