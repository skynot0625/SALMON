from typing import Any, Dict, Tuple
import functools  # 이 라인을 추가하세요
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp2_model import IntegratedResNet
from aihwkit.optim import AnalogSGD, AnalogAdam
from aihwkit.nn.conversion import convert_to_analog
from aihwkit.simulator.presets import TikiTakaEcRamPreset, IdealizedPreset, EcRamPreset
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, IdealDevice
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
import hydra
import wandb

class SalmonLitModule(LightningModule):
    def __init__(
        self,
        model: str,
        integrated_resnet: IntegratedResNet,
        optimizer: dict,
        compile: bool,
        dataset: str,
        epoch: int,
        dataset_path: str,
        autoaugment: bool,
        batchsize: int,
        N_CLASSES: int,
        opt_config : str,
        sch_config : str
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize only the backbone component from the integrated_resnet
        self.backbone = integrated_resnet.backbone

        # Store additional parameters as needed
        self.compile = compile

        # Optimizer and scheduler settings
        self.optimizer_config = optimizer
#         self.hparams.architecture = integrated_resnet.architecture
#         self.hparams.num_classes = integrated_resnet.num_classes
#         self.hparams.rpu_config = {
#             '_target_': integrated_resnet.rpu_config.__class__.__module__ + "." + integrated_resnet.rpu_config.__class__.__qualname__,
            # Add additional properties of rpu_config as needed
#         }
        # Initialize metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.val_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.test_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        
    def forward(self, x):
        # 단순히 backbone 모델을 사용하여 출력을 계산합니다.
        out_backbone, _, _, _, _ ,_ = self.backbone(x)
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


    # Validation step, test step, configure optimizers, etc.


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
        # AnalogSGD로 최적화기 설정 변경
        optimizer = AnalogSGD(self.parameters(), lr=self.hparams.optimizer['lr'],
                            weight_decay=self.hparams.optimizer['weight_decay'],
                            momentum=self.hparams.optimizer.get('momentum', 0),  # momentum 추가, 기본값은 0으로 설정
                            dampening=self.hparams.optimizer.get('dampening', 0),  # dampening 추가, 기본값은 0으로 설정
                            nesterov=self.hparams.optimizer.get('nesterov', False))  # nesterov 추가, 기본값은 False로 설정
        optimizer.regroup_param_groups(self.parameters())
        
        # 스케줄러를 사용하지 않으므로, 최적화기만 반환
        return optimizer

    
if __name__ == "__main__":
    _ = SalmonLitModule(None, None, None, None)