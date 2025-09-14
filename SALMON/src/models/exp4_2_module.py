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
        p_max: int,
        opt_config : str,
        sch_config : str
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize IntegratedResNet with parameters from integrated_resnet_config
        
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

        # loss function and metrics 초기화
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_0_best = MaxMetric()
        self.adaptation_layers = torch.nn.ModuleList()
        self.init_adaptation_layers = False
        
    def setup_adaptation_layers(self, feature_sizes):
        for student_size in feature_sizes:
            self.adaptation_layers.append(torch.nn.Linear(student_size, feature_sizes[0]).to(self.device))

    def forward(self, x):
        # Forward pass through the backbone
        out_backbone, feature_backbone, x1, x2, x3 = self.backbone(x)

        # Forward pass through the attention mechanisms
        out_attention1, feature_attention1 = self.attention1(x1)
        out_attention2, feature_attention2 = self.attention2(x2)
        out_attention3, feature_attention3 = self.attention3(x3)

        # Combine outputs and features from different layers
        outputs = [out_backbone, out_attention3, out_attention2, out_attention1]
        features = [feature_backbone, feature_attention3, feature_attention2, feature_attention1]

        # Return all necessary outputs and features for further processing
        return outputs, features

    def model_step(self, batch):
        inputs, labels = batch
        inputs = inputs.to(self.device)

        # Forward pass for features and outputs
        out_backbone, feature_backbone, x1, x2, x3 = self.backbone(inputs)
        out_attention1, feature_attention1 = self.attention1(x1)
        out_attention2, feature_attention2 = self.attention2(x2)
        out_attention3, feature_attention3 = self.attention3(x3)

        # Combine outputs and features from different layers
        outputs = [out_backbone, out_attention3, out_attention2, out_attention1]
        features = [feature_backbone, feature_attention3, feature_attention2, feature_attention1]

        # Return all necessary outputs and features along with labels for loss computation
        return outputs, features, labels

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # Forward pass for features and outputs
        out_backbone, feature_backbone, x1, x2, x3 = self.backbone(inputs)
        out_attention1, feature_attention1 = self.attention1(x1)
        out_attention2, feature_attention2 = self.attention2(x2)
        out_attention3, feature_attention3 = self.attention3(x3)

        outputs = [out_backbone, out_attention3, out_attention2, out_attention1]
        features = [feature_backbone, feature_attention3, feature_attention2, feature_attention1]

        # Initialize adaptation layers if not done
        if not self.init_adaptation_layers:
            self.setup_adaptation_layers([f.size(1) for f in features])
            self.init_adaptation_layers = True

        # Compute self-distillation loss
        loss = self.criterion(outputs[0], labels)
        teacher_output = outputs[0].detach()
        teacher_feature = features[0].detach()

        for idx, (output, feature) in enumerate(zip(outputs[1:], features[1:])):
            # Logits distillation
            loss += self.cross_entropy_distillation(output, teacher_output) * self.loss_coefficient
            loss += self.criterion(output, labels) * (1 - self.loss_coefficient)

            # Feature distillation
            if idx != 0:
                loss += torch.dist(self.adaptation_layers[idx-1](feature), teacher_feature) * self.feature_loss_coefficient

        # 메트릭 업데이트 및 로깅
        self.train_acc(torch.argmax(outputs[0], dim=1), labels)
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
        outputs, features, labels = self.model_step(batch)
        total_loss = 0
        total_acc = []

        # 각 네트워크에 대한 손실 및 정확도 계산 및 로깅
        for i, output in enumerate(outputs):
            loss = self.criterion(output, labels)
            acc = self.val_acc(torch.argmax(output, dim=1), labels)
            total_loss += loss
            total_acc.append(acc)

            self.log(f"val/loss_{i}", loss, on_step=False, on_epoch=True)
            self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True)
            if i == 0:  # 첫 번째 네트워크에 대한 정확도를 별도로 추적
                self.val_acc_0_best.update(acc)

        avg_loss = total_loss / len(outputs)
        avg_acc = sum(total_acc) / len(total_acc)
        return {"val_loss": avg_loss, "val_acc": avg_acc}

    def on_validation_epoch_end(self):
        # 첫 번째 네트워크의 최고 검증 정확도를 로깅
        val_acc_0_best = self.val_acc_0_best.compute()
        self.log("val/acc_0_best", val_acc_0_best, prog_bar=True)
        
        self.val_acc.reset()  # 모든 정확도 추적기를 리셋
        self.val_acc_0_best.reset()


    def test_step(self, batch, batch_idx):
        outputs, features, labels = self.model_step(batch)
        main_output = outputs[0]  # 메인 출력을 사용하여 손실과 정확도를 계산합니다.

        loss = self.criterion(main_output, labels)
        acc = self.test_acc(torch.argmax(main_output, dim=1), labels)

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
        # parameters = list(self.backbone.parameters()) + \
        #             list(self.attention1.parameters()) + \
        #             list(self.attention2.parameters()) + \
        #             list(self.attention3.parameters())
        
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