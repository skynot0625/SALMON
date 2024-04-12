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

def load_checkpoint(checkpoint_path, model_component, prefix_remove=None):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']

    # 키 이름 조정: 사전 훈련된 모델의 키 이름에서 특정 접두어 제거
    if prefix_remove:
        new_state_dict = {key.replace(prefix_remove, ''): value for key, value in state_dict.items() if key.startswith(prefix_remove)}
    else:
        new_state_dict = state_dict

    # 모델에 상태 사전 로드
    model_component.load_state_dict(new_state_dict)  # strict=False를 사용하여 일치하지 않는 키 무시


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
        autoaugment: bool,
        checkpoint_path: str = None,
        temperature: float = 3.0
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize only the backbone component from the integrated_resnet
        self.student = integrated_resnet.backbone
        self.teacher = integrated_resnet_t.backbone
        if checkpoint_path is not None:
            # 'backbone.' 접두어를 제거하도록 설정
            load_checkpoint(checkpoint_path, self.teacher, prefix_remove='backbone.')

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
        self.temperature = temperature
#         self.hparams.architecture = integrated_resnet.architecture
#         self.hparams.num_classes
    def forward(self, x):
        return self.student(x)

    def model_step(self, batch):
        inputs, labels = batch
        # 학생 모델에서 반환된 모든 값을 처리합니다.
        out4_s, feature_s, x4_s, x1_s, x2_s, x3_s = self.student(inputs)
        # 교사 모델에서 반환된 모든 값을 처리합니다.
        out4_t, feature_t, x4_t, x1_t, x2_t, x3_t = self.teacher(inputs)

        return out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, labels, x1_s, x2_s, x3_s, x1_t, x2_t, x3_t


    def soft_target_loss(self, student_logits, teacher_logits):
        """
        Compute the SoftTarget loss between student and teacher outputs.
        """
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (self.temperature ** 2)

    def training_step(self, batch, batch_idx):
        # Extract data
        inputs, labels = batch
        out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, labels, x1_s, x2_s, x3_s, x1_t, x2_t, x3_s = self.model_step(inputs)

        # Compute the standard cross-entropy loss
        ce_loss = self.criterion(out4_s, labels)

        # Compute the knowledge distillation loss using the soft_target_loss method
        st_loss = self.soft_target_loss(out4_s, out4_t)

        # Combine the losses
        total_loss = ce_loss + self.lambda_kd * st_loss

        # Logging
        self.log("train/ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/st_loss", st_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.train_acc(pred, labels)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss


    def validation_step(self, batch, batch_idx):
        # Correctly unpack values returned by model_step
        out4_s, feature_s, x4_s, out4_t, feature_t, x4_t, labels, x1_s, x2_s, x3_s, x1_t, x2_t, x3_t = self.model_step(batch)

        # Use out4_s for the classification loss
        student_loss = self.criterion(out4_s, labels)
        self.log("val/student_loss", student_loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.val_acc(pred, labels)
        self.log("val/student_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.train_teacher:
            # If training the teacher as well, compute and log its loss and accuracy
            teacher_loss = self.criterion(out4_t, labels)
            self.log("val/teacher_loss", teacher_loss, on_step=False, on_epoch=True, prog_bar=True)
            teacher_acc = self.val_acc(torch.argmax(out4_t, dim=1), labels)
            self.log("val/teacher_acc", teacher_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"student_loss": student_loss, "student_acc": acc}

    def test_step(self, batch, batch_idx):
        # Correctly unpack values returned by model_step for the test scenario
        out4_s, _, _, _, _, _, labels, _, _, _, _, _, _ = self.model_step(batch)

        # Compute the classification loss
        loss = self.criterion(out4_s, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Compute and log the accuracy
        pred = torch.argmax(out4_s, dim=1)
        acc = self.test_acc(pred, labels)
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