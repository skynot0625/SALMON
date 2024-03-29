from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import Accuracy, MaxMetric, MeanMetric
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from src.models.components.salmon import IntegratedResNet  # IntegratedResNet 클래스 경로 수정
  # AnalogLeNet 클래스 경로 수정

class CIFAR10LitModule(LightningModule):
    """LightningModule for training, validating, and testing an AnalogLeNet model on the CIFAR-10 dataset."""

    def __init__(
        self,
        net: IntegratedResNet,  # 모델 타입 변경
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        batch_size: int = 64,
        num_workers: int = 4
    ) -> None:
        """Initialize a `CIFAR10LitModule`.

        :param net: The model to train (AnalogLeNet).
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        :param compile: Whether to compile the model for PyTorch 2.0.
        :param batch_size: Batch size for data loaders.
        :param num_workers: Number of workers for data loaders.
        """
        super().__init__()
        self.save_hyperparameters()

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        # Define metrics
        self.train_acc = Accuracy(num_classes=10)
        self.val_acc = Accuracy(num_classes=10)
        self.test_acc = Accuracy(num_classes=10)

        # Define metrics for averaging loss
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Metric for tracking best validation accuracy
        self.val_acc_best = MaxMetric()

        # Data-related members
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Download CIFAR-10 data and prepare it for training."""
        datasets.CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
        datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())

    def setup(self, stage: str = None):
        """Set up the data for training/validation/testing."""
        if stage == "fit" or stage is None:
            cifar_full = datasets.CIFAR10(root="./data", train=True, transform=transforms.ToTensor())
            self.train_dataset, self.val_dataset = random_split(cifar_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.test_dataset = datasets.CIFAR10(root="./data", train=False, transform=transforms.ToTensor())

        if self.hparams.compile and stage == "fit":
            # PyTorch 2.0 compile option
            self.net = torch.compile(self.net)

    def train_dataloader(self):
        """Create the training data loader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Create the validation data loader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """Create the test data loader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose and configure the optimizer and learning-rate scheduler.

        :return: A dict containing the configured optimizer and learning-rate scheduler.
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1
            }
        }