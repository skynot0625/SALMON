from typing import Any, Dict, Tuple
import functools  # 이 라인을 추가하세요
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp2_model import IntegratedResNet, IntegratedResNet_T
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
        student: IntegratedResNet,
        teacher : IntegratedResNet_T,
        optimizer: dict,
        compile: bool,
        dataset: str,
        epoch: int,
        dataset_path: str,
        batchsize: int,
        N_CLASSES: int,
        opt_config : str,
        sch_config : str,
        temperature: float,
        p: float,
        lambda_kd: float,  # 추가된 파라미터
        train_teacher : bool
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize only the backbone component from the integrated_resnet
        self.student = student.backbone
        self.teacher = teacher.backbone

        # Store additional parameters as needed
        self.compile = compile
        self.temperature = temperature
        self.p = p
        self.lambda_kd = lambda_kd  # lambda_kd를 클래스 변수로 저장
        self.train_teacher = train_teacher
        # Optimizer and scheduler settings
        self.optimizer_config = optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.val_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.test_acc = Accuracy(task="multiclass", num_classes=N_CLASSES)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
#         self.hparams.architecture = integrated_resnet.architecture
#         self.hparams.num_classes
    def forward(self, x):
        return self.student(x)

    def model_step(self, batch):
        inputs, labels = batch
        student_outputs = self.student(inputs)
        teacher_outputs = self.teacher(inputs) if self.hparams.train_teacher else None
        return student_outputs, teacher_outputs, labels

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        student_outputs, teacher_outputs, _ = self.model_step(batch)

        # Compute the cross-entropy loss for the student model
        ce_loss_student = self.criterion(student_outputs, labels)

        # Initialize the teacher loss to zero if not training the teacher
        ce_loss_teacher = 0.0
        if self.hparams.train_teacher:
            # Compute the cross-entropy loss for the teacher model, if we are training it
            ce_loss_teacher = self.criterion(teacher_outputs, labels)

        # Compute the Attention Transfer (AT) loss, if the teacher model is used
        at_loss = self.attention_transfer_loss(student_outputs, teacher_outputs) if teacher_outputs is not None else 0.0

        # Combine the losses
        # If train_teacher is True, both the student's and teacher's losses are included
        # along with the AT loss. If False, only the student's loss and AT loss are considered.
        total_loss = ce_loss_student + self.hparams.lambda_kd * at_loss + ce_loss_teacher

        # Logging
        self.log("train/loss_student", ce_loss_student, on_step=False, on_epoch=True, prog_bar=True)
        if self.hparams.train_teacher:
            self.log("train/loss_teacher", ce_loss_teacher, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_at", at_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)

        acc = self.train_acc(torch.argmax(student_outputs, dim=1), labels)
        self.log("train/acc_student", acc, on_step=False, on_epoch=True, prog_bar=True)

        # To ensure the optimizer applies the gradients, return the total loss
        return total_loss


    def attention_transfer_loss(self, student_outputs, teacher_outputs, p=2):
        def attention_map(features, p):
            am = features.pow(p)
            am = am.mean(1, keepdim=True)
            am = am / am.sum([2, 3], keepdim=True)
            return am

        student_am = attention_map(student_outputs, p)
        teacher_am = attention_map(teacher_outputs, p)
        return F.mse_loss(student_am, teacher_am)

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        student_outputs, teacher_outputs, _ = self.model_step(batch)
        loss = self.criterion(student_outputs, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.val_acc(torch.argmax(student_outputs, dim=1), labels)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        student_outputs, _, _ = self.model_step(batch)
        loss = self.criterion(student_outputs, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = self.test_acc(torch.argmax(student_outputs, dim=1), labels)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)


    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.val_acc.reset()

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
       optimizer = AnalogSGD(
            list(self.student.parameters()) + list(self.teacher.parameters()), 
            lr=self.hparams.optimizer['lr'],
            weight_decay=self.hparams.optimizer['weight_decay'],
            momentum=self.hparams.optimizer.get('momentum', 0),
            dampening=self.hparams.optimizer.get('dampening', 0),
            nesterov=self.hparams.optimizer.get('nesterov', False)
        )
        
        optimizer.regroup_param_groups(self.student)
        
        # 스케줄러를 사용하지 않으므로, 최적화기만 반환
        return optimizer

    
if __name__ == "__main__":
    _ = SalmonLitModule(None, None, None, None)