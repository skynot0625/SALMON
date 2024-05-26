from typing import Any, Dict, Tuple
import functools  # 이 라인을 추가하세요
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp7_1_model import IntegratedResNet
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
from aihwkit.simulator.configs import SingleRPUConfig, ConstantStepDevice, IdealDevice
from aihwkit.simulator.configs  import SoftBoundsDevice, SoftBoundsPmaxDevice

class SalmonLitModule(LightningModule):
    def __init__(
        self,
        model: str,
        integrated_resnet: IntegratedResNet,
        compile: bool,
        optimizer: dict,
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
        opt_config: str,
        sd_config: str,
        FC_Digit: str,
        sch_config: str,
        scheduler: dict
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize components from integrated_resnet
        self.input = integrated_resnet.input_module
        self.features = integrated_resnet.features
        self.classifier = integrated_resnet.classifier
        self.attention1 = integrated_resnet.attention1
        self.attention2 = integrated_resnet.attention2
        self.attention3 = integrated_resnet.attention3

        # Initialize other properties and metrics
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
        self.opt_config = opt_config
        self.sd_config = sd_config
        self.FC_Digit = FC_Digit

        # Initialize loss functions and metrics
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

# 이 구성을 통해 생성자에 `scheduler`를 전달하는 문제를 해결하고, 설정은 `configure_optimizers`에서 관리됩니다.

    
    # def on_train_epoch_end(self, unused=None, outputs=None):
    #     # 모든 레이어 목록 정의
    #     layers = {
    #         'input': self.input,
    #         'features_layer1': self.features.layer1,
    #         'features_layer2': self.features.layer2,
    #         'features_layer3': self.features.layer3,
    #         'features_layer4': self.features.layer4,
    #         'classifier': self.classifier
    #     }

    #     # 각 레이어의 가중치 평균과 표준 편차 로깅
    #     for layer_name, layer in layers.items():
    #             weight_dicts = layer.get_weights()  # 가중치 가져오기
    #             for sub_layer_name, (weight, bias) in weight_dicts.items():
    #                 weight_mean = weight.mean().item()
    #                 weight_std = weight.std().item()

    #                 # 로깅
    #                 self.log(f'{layer_name}_{sub_layer_name}/weight_mean', weight_mean)
    #                 self.log(f'{layer_name}_{sub_layer_name}/weight_std', weight_std)

    #                 # 터미널 출력
    #                 # print(f"{layer_name}_{sub_layer_name} - Weight Mean: {weight_mean}, Std: {weight_std}")


    def setup_adaptation_layers(self, feature_sizes):
        for student_size in feature_sizes:
            self.adaptation_layers.append(torch.nn.Linear(student_size, feature_sizes[0]).to(self.device))

    def forward(self, x):
        # Forward pass through the 
        input_1 = self.input(x)
        feature_backbone, x1, x2, x3 = self.features(input_1)
        out_backbone = self.classifier(feature_backbone)

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
        input_1 = self.input(inputs)
        feature_backbone, x1, x2, x3 = self.features(input_1)

        # Classification step
        out_backbone = self.classifier(feature_backbone)

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

        # Use the model_step to forward inputs through the network
        outputs, features, _ = self.model_step((inputs, labels))

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
#             loss += self.cross_entropy_distillation(teacher_output,output) * self.loss_coefficient
            loss += self.criterion(output, labels) * (1-self.loss_coefficient)

            # Feature distillation for subsequent layers
#             if idx != 0:
#                 loss += torch.dist(self.adaptation_layers[idx-1](feature), teacher_feature) * self.feature_loss_coefficient

        # Metrics update and logging
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

            # 모든 네트워크 출력에 대해 일관된 로깅 인자 사용
            self.log(f"val/loss_{i}", loss, on_step=False, on_epoch=True)
            if i == 0:
                # 첫 번째 출력에 대한 정확도만 별도로 로깅 (progress bar 추가)
                self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True)

        avg_loss = total_loss / len(outputs)
        avg_acc = sum(total_acc) / len(total_acc)
        return {"val_loss": avg_loss, "val_acc": avg_acc}


    def on_validation_epoch_end(self):
        self.val_acc.reset()  # 모든 정확도 추적기를 리셋



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
        optimizer = AnalogSGD(self.parameters(), lr=self.hparams.optimizer['lr'],
                              weight_decay=self.hparams.optimizer['weight_decay'],
                              momentum=self.hparams.optimizer.get('momentum', 0.9),
                              dampening=self.hparams.optimizer.get('dampening', 0),
                              nesterov=self.hparams.optimizer.get('nesterov', False))
#step lr      
#         scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                     step_size=self.hparams.scheduler['step_size'],
#                                                     gamma=self.hparams.scheduler['gamma'])
#         scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
    # Cosine Annealing scheduler 설정, 기본값을 사용
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=self.hparams.scheduler.get('T_max', 300),
                                                               eta_min=self.hparams.scheduler.get('eta_min', 0.0001))

        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}

        return [optimizer], [scheduler_config]




    

if __name__ == "__main__":
    _ = SalmonLitModule(None, None, None, None)