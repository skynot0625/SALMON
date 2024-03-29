from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp3_model import IntegratedResNet
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
from aihwkit.simulator.configs import SoftBoundsPmaxDevice
from aihwkit.simulator.configs import SingleRPUConfig
from aihwkit.simulator.configs  import SoftBoundsDevice, SoftBoundsPmaxDevice

class SalmonLitModule(LightningModule):
    def __init__(
        self,
        model: str,
        integrated_resnet: IntegratedResNet,  # integrated_resnet_config 대신 integrated_resnet 사 # Changed from rpu_config to integrated_resnet_config
        optimizer: dict,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        dataset: str,
        epoch: int,
        loss_coefficient: float,
        feature_loss_coefficient: float,
        dataset_path: str,
        autoaugment: bool,
        temperature: float,
        batchsize: int,
        init_lr: float,
        N_CLASSES: int,
        block: str,
        alpha: float,
        p_max: int
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # 모듈 구성요소 설정
        self.backbone = integrated_resnet.backbone
        self.attention1 = integrated_resnet.attention1
        self.attention2 = integrated_resnet.attention2
        self.attention3 = integrated_resnet.attention3

        # 기타 매개변수들을 클래스 속성으로 저장
        self.compile = compile
        self.model = model
        self.dataset = dataset
        self.epoch = epoch
        self.loss_coefficient = loss_coefficient
        self.feature_loss_coefficient = feature_loss_coefficient
        self.dataset_path = dataset_path
        self.autoaugment = autoaugment
        self.temperature = temperature
        self.batchsize = batchsize
        self.init_lr = init_lr
        self.N_CLASSES = N_CLASSES
        self.block = block
        self.alpha = alpha
        self.p_max = p_max

        # 옵티마이저와 스케줄러 설정
        self.optimizer_config = optimizer
        self.scheduler = scheduler

        # loss function and metrics 초기화
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        
    def forward(self, x):
        # 단순히 backbone 모델을 사용하여 출력을 계산합니다.
        out_backbone, _, _, _, _ = self.backbone(x)
        return out_backbone

    def model_step(self, batch):
        inputs, labels = batch
        outputs = self(inputs)
        return outputs, labels

    def training_step(self, batch, batch_idx):
        outputs, labels = self.model_step(batch)
        loss = self.criterion(outputs, labels)

        self.train_acc(torch.argmax(outputs, dim=1), labels)
        self.train_loss(loss)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def cross_entropy_distillation(self, student_output, teacher_output):
        log_softmax_outputs = F.log_softmax(student_output / self.temperature, dim=1)
        softmax_targets = F.softmax(teacher_output / self.temperature, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()


    def validation_step(self, batch, batch_idx):
        outputs, labels = self.model_step(batch)
        loss = self.criterion(outputs, labels)
        acc = self.val_acc(torch.argmax(outputs, dim=1), labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        outputs, labels = self.model_step(batch)
        loss = self.criterion(outputs, labels)
        acc = self.test_acc(torch.argmax(outputs, dim=1), labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)
        return {"test_loss": loss, "test_acc": acc}

    def on_test_epoch_end(self):
        # 여기서 필요한 추가적인 작업을 수행할 수 있습니다.
        pass
    

    def setup(self, stage: str) -> None:
        """모델의 초기화 및 설정을 위한 메서드."""
        # 여기서 필요한 초기화 또는 설정을 수행합니다.
        # 예를 들어, 모델의 가중치를 불러오거나, 특정 조건에 따라 모델 구성을 변경할 수 있습니다.
        if stage == "fit":
            pass
            #  self.net = torch.compile(self.net)
            # 훈련 단계에서만 특정 작업을 수행합니다.
            # 예: self.backbone = self.load_pretrained_backbone()
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.backbone.parameters(), **self.optimizer_config)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.scheduler.T_max, eta_min=self.hparams.scheduler.eta_min),
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]


if __name__ == "__main__":
    _ = SalmonLitModule(None, None, None, None)
