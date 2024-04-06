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
        integrated_resnet: IntegratedResNet,
        integrated_resnet_t : IntegratedResNet_T,
        optimizer: dict,
        compile: bool,
        dataset: str,
        epoch: int,
        dataset_path: str,
        batchsize: int,
        N_CLASSES: int,
        opt_config : str,
        sch_config : str,
        p: float,
        lambda_kd: float,  # 추가된 파라미터
        train_teacher : bool,
        autoaugment: bool
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize only the backbone component from the integrated_resnet
        self.student = integrated_resnet.backbone
        self.teacher = integrated_resnet_t.backbone

        # Store additional parameters as needed
        self.compile = compile
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
        self.autoaugment = autoaugment
#         self.hparams.architecture = integrated_resnet.architecture
#         self.hparams.num_classes
    def forward(self, x):
        return self.student(x)

    def model_step(self, batch):
        inputs, labels = batch
        # 학생 모델에서 반환된 모든 값을 처리합니다.
        out4_s, feature_s, x4_s, _, _, _ = self.student(inputs)  # 추가된 반환 값을 무시합니다.
            # 교사 모델에서 반환된 모든 값을 처리합니다.
        out4_t, feature_t, x4_t, _, _, _ = self.teacher(inputs)  # 추가된 반환 값을 무시합니다.

        return out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, labels

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        # Extract outputs and the last feature map (before pooling) from student and teacher networks
        out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, _ = self.model_step(batch)

        # Compute the cross-entropy loss for the student's output
        ce_loss = self.criterion(out4_s, labels)

        # Initialize AT loss
        at_loss = 0.0
            # Compute the AT loss between student and teacher last feature maps (x4)
        at_loss = self.attention_transfer_loss(x4_s, x4_t, self.p)

        # Combine the cross-entropy loss and the AT loss
        total_loss = ce_loss + self.lambda_kd * at_loss

        # Logging
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/at_loss", at_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.train_acc(pred, labels)  # 수정된 부분
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def attention_transfer_loss(self, student_features, teacher_features, p):
        """
        Computes the Attention Transfer loss between student and teacher features.
        """
        student_am = self.attention_map(student_features, p)
        teacher_am = self.attention_map(teacher_features, p)
        return F.mse_loss(student_am, teacher_am)

    def attention_map(self, fm, p):
        """
        Generates an attention map from the given feature map.
        """
        am = torch.pow(torch.abs(fm), p)
        am = torch.sum(am, dim=1, keepdim=True)  #Sum  over channels
        norm = torch.norm(am, p='fro', dim=[2, 3], keepdim=True)  # Normalize over spatial dimensions
        am = am / (norm + 1e-6)
        return am

    def validation_step(self, batch, batch_idx):
        out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, labels = self.model_step(batch)

        # Use out4_s for the classification loss
        student_loss = self.criterion(out4_s, labels)
        self.log("val/student_loss", student_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.val_acc(pred, labels)  # 수정된 부분
        self.log("val/student_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.train_teacher and out4_t is not None:
            teacher_loss = self.criterion(out4_t, labels)
            self.log("val/teacher_loss", teacher_loss, on_step=False, on_epoch=True, prog_bar=True)
            teacher_acc = self.val_acc(torch.argmax(out4_t, dim=1), labels)  # 수정된 부분
            self.log("val/teacher_acc", teacher_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"student_loss": student_loss, "student_acc": acc}

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        out4_s, _, _ = self.model_step(batch)[:3]

        # Compute the classification loss
        loss = self.criterion(out4_s, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.test_acc(pred, labels)  # 수정된 부분
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