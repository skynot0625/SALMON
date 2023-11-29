# src/analog_lenet.py 파일에 들어갈 내용

from torch import nn
from src.aihwkit.nn import AnalogSequential, AnalogConv2d, AnalogLinear
from src.aihwkit.simulator.configs import InferenceRPUConfig

class AnalogLeNet(nn.Module):
    """LeNet-스타일의 아날로그 컨볼루션 신경망."""

    def __init__(self, rpu_config: InferenceRPUConfig):
        """신경망 초기화.

        Args:
            rpu_config (InferenceRPUConfig): RPU 설정.
        """
        super(AnalogLeNet, self).__init__()

        # Define the number of channels and other parameters used in the model.
        channel = [16, 32, 512, 128]
        N_CLASSES = 10  # Number of output classes

        # 네트워크 구성
        self.model = AnalogSequential(
            AnalogConv2d(in_channels=1, out_channels=channel[0], kernel_size=5, stride=1, rpu_config=rpu_config),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            AnalogConv2d(in_channels=channel[0], out_channels=channel[1], kernel_size=5, stride=1, rpu_config=rpu_config),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            AnalogLinear(in_features=channel[2], out_features=channel[3], rpu_config=rpu_config),
            nn.Tanh(),
            AnalogLinear(in_features=channel[3], out_features=N_CLASSES, rpu_config=rpu_config),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.Tensor): 입력 텐서.

        Returns:
            torch.Tensor: 출력 텐서.
        """
        return self.model(x)
