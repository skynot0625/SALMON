from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.salmon import IntegratedResNet


class SalmonLitModule(LightningModule):
    def __init__(
        self,
        integrated_resnet: IntegratedResNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        model=None,  # args 항목 추가
        dataset=None,
        epoch=None,
        loss_coefficient=None,
        feature_loss_coefficient=None,
        dataset_path=None,
        autoaugment=None,
        temperature=None,
        batchsize=None,
        init_lr=None,
        N_CLASSES=None,
        block=None,
        alpha=None,
        p_max=None
    ):
        super().__init__()
        self.save_hyperparameters()

        # args 항목들을 클래스 속성으로 저장
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

        # IntegratedResNet 인스턴스에서 모듈을 가져옵니다
        self.backbone = integrated_resnet.backbone
        self.attention1 = integrated_resnet.attention1
        self.attention2 = integrated_resnet.attention2
        self.attention3 = integrated_resnet.attention3

        # 여기에 나머지 초기화 코드를 추가합니다

        # loss function and metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_acc = Accuracy(task="multiclass", num_classes=10)
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

        # Optimizer and Scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler


    def forward(self, x):
        out_backbone, _, x1, x2, x3 = self.backbone(x)
        out1, _ = self.attention1(x1)
        out2, _ = self.attention2(x2)
        out3, _ = self.attention3(x3)
        return out1, out2, out3, out_backbone

    def model_step(self, batch):
        inputs, labels = batch
        out1, out2, out3, out_backbone = self(inputs)
        return out1, out2, out3, out_backbone, labels

    def training_step(self, batch, batch_idx):
        out1, out2, out3, out_backbone, labels = self.model_step(batch)

        loss = self.criterion(out_backbone, labels)
        for output in [out1, out2, out3]:
            loss += self.criterion(output, labels)
            loss += self.cross_entropy_distillation(output, out_backbone.detach())

        self.train_acc(torch.argmax(out_backbone, dim=1), labels)
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
        images, labels = batch
        out1, out2, out3, out_backbone = self(images)

        loss = self.criterion(out_backbone, labels)
        preds = torch.argmax(out_backbone, dim=1)
        acc = self.val_acc(preds, labels)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.val_acc.reset()

    def test_step(self, batch, batch_idx):
        images, labels = batch
        out1, out2, out3, out_backbone = self(images)

        loss = self.criterion(out_backbone, labels)
        preds = torch.argmax(out_backbone, dim=1)
        acc = self.test_acc(preds, labels)
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
            
    def configure_optimizers(self) -> Dict[str, Any]:
        optimizers = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizers)
            return {
                "optimizer": optimizers,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizers}


if __name__ == "__main__":
    _ = SalmonLitModule(None, None, None, None)
