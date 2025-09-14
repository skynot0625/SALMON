from typing import Any, Dict, Tuple
import functools
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from src.models.components.exp7_6_digital_model import IntegratedResNet
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
        scheduler: dict,
        # New parameters for 6-head configuration
        use_all_heads: bool = True,  # Whether to use all 6 heads or just original 3
        head_weights: list = None,    # Optional weights for each head in ensemble
        active_heads: list = None,    # Which heads to use: e.g., ['D1','D2','D3','D6'] or None for all
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the integrated model
        self.integrated_resnet = integrated_resnet
        
        # Store configuration
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
        self.use_all_heads = use_all_heads
        self.active_heads = active_heads
        
        # Determine which heads are active
        if active_heads is not None:
            # Custom head selection
            self.head_names = ['backbone'] + active_heads
            self.num_heads = len(self.head_names)
        elif use_all_heads:
            # All 6 attention heads
            self.head_names = ['backbone', 'D1', 'D4', 'D2', 'D5', 'D3', 'D6']
            self.num_heads = 7
        else:
            # Original 3 heads
            self.head_names = ['backbone', 'D3', 'D2', 'D1']
            self.num_heads = 4
        
        # Initialize head weights for ensemble
        if head_weights is None:
            # Default equal weights for all active heads
            self.head_weights = [1.0] * self.num_heads
        else:
            self.head_weights = head_weights

        # Initialize loss functions and metrics
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Metrics for tracking each head's performance
        self.train_acc_heads = torch.nn.ModuleList([
            Accuracy(task="multiclass", num_classes=N_CLASSES) 
            for _ in range(self.num_heads)
        ])
        self.val_acc_heads = torch.nn.ModuleList([
            Accuracy(task="multiclass", num_classes=N_CLASSES) 
            for _ in range(self.num_heads)
        ])
        self.test_acc_heads = torch.nn.ModuleList([
            Accuracy(task="multiclass", num_classes=N_CLASSES) 
            for _ in range(self.num_heads)
        ])
        
        # Overall metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.val_acc_best = MaxMetric()
        
        # Adaptation layers for feature distillation
        self.adaptation_layers = torch.nn.ModuleList()
        self.init_adaptation_layers = False

    def setup_adaptation_layers(self, feature_sizes):
        """Setup adaptation layers for feature alignment in distillation."""
        teacher_size = feature_sizes[0]  # Backbone feature size
        for student_size in feature_sizes[1:]:
            if student_size != teacher_size:
                self.adaptation_layers.append(
                    torch.nn.Linear(student_size, teacher_size).to(self.device)
                )
            else:
                self.adaptation_layers.append(torch.nn.Identity())

    def forward(self, x):
        """Forward pass using all heads or original configuration."""
        if self.use_all_heads:
            return self.integrated_resnet.forward_all_heads(x)
        else:
            return self.integrated_resnet.forward(x)

    def cross_entropy_distillation(self, student_output, teacher_output):
        """Knowledge distillation loss between student and teacher outputs."""
        log_softmax_outputs = F.log_softmax(student_output / self.temperature, dim=1)
        softmax_targets = F.softmax(teacher_output / self.temperature, dim=1)
        return -(log_softmax_outputs * softmax_targets).sum(dim=1).mean()

    def model_step_all_heads(self, batch):
        """Model step for 6-head configuration with selective head activation."""
        inputs, labels = batch
        inputs = inputs.to(self.device)
        
        # Get all 6 attention heads + backbone
        outputs = self.integrated_resnet.forward_all_heads(inputs)
        
        # Parse outputs
        out_backbone = outputs[0]
        out4_feature = outputs[1]
        
        # Attention head outputs and features (D1-D6 in interleaved order)
        D1_out, D1_feat = outputs[2], outputs[3]
        D4_out, D4_feat = outputs[4], outputs[5]
        D2_out, D2_feat = outputs[6], outputs[7]
        D5_out, D5_feat = outputs[8], outputs[9]
        D3_out, D3_feat = outputs[10], outputs[11]
        D6_out, D6_feat = outputs[12], outputs[13]
        
        # Map head names to outputs and features
        head_outputs_map = {
            'backbone': out_backbone,
            'D1': D1_out, 'D2': D2_out, 'D3': D3_out,
            'D4': D4_out, 'D5': D5_out, 'D6': D6_out
        }
        head_features_map = {
            'backbone': out4_feature,
            'D1': D1_feat, 'D2': D2_feat, 'D3': D3_feat,
            'D4': D4_feat, 'D5': D5_feat, 'D6': D6_feat
        }
        
        # Select only active heads
        selected_outputs = []
        selected_features = []
        for head_name in self.head_names:
            if head_name in head_outputs_map:
                selected_outputs.append(head_outputs_map[head_name])
                selected_features.append(head_features_map[head_name])
        
        return selected_outputs, selected_features, labels

    def model_step_original(self, batch):
        """Model step for original 3-head configuration."""
        inputs, labels = batch
        inputs = inputs.to(self.device)
        
        # Get original 3 attention heads + backbone
        outputs = self.integrated_resnet.forward(inputs)
        
        # Parse outputs (original forward returns 8 values)
        out1, feature1, out2, feature2, out3, feature3, out4, out4_feature = outputs
        
        # Order: [backbone, D3, D2, D1] (reverse order for consistency with training)
        all_outputs = [out4, out3, out2, out1]
        all_features = [out4_feature, feature3, feature2, feature1]
        
        return all_outputs, all_features, labels

    def training_step(self, batch, batch_idx):
        """Training step with self-distillation for all heads."""
        # Get outputs based on configuration
        if self.use_all_heads:
            outputs, features, labels = self.model_step_all_heads(batch)
        else:
            outputs, features, labels = self.model_step_original(batch)
        
        labels = labels.to(self.device)
        
        # Initialize adaptation layers if needed
        if not self.init_adaptation_layers:
            self.setup_adaptation_layers([f.size(1) for f in features])
            self.init_adaptation_layers = True
        
        # Compute self-distillation loss
        # Backbone (teacher) loss
        loss = self.criterion(outputs[0], labels)
        
        # Teacher outputs for distillation
        teacher_output = outputs[0].detach()
        teacher_feature = features[0].detach()
        
        # Distillation from attention heads
        for idx, (output, feature) in enumerate(zip(outputs[1:], features[1:])):
            # Classification loss for each head
            loss += self.criterion(output, labels) * (1 - self.loss_coefficient) * self.head_weights[idx+1]
            
            # Knowledge distillation loss
            loss += self.cross_entropy_distillation(output, teacher_output) * self.loss_coefficient * self.head_weights[idx+1]
            
            # Feature distillation (optional)
            if self.feature_loss_coefficient > 0 and idx < len(self.adaptation_layers):
                adapted_feature = self.adaptation_layers[idx](feature)
                loss += torch.dist(adapted_feature, teacher_feature) * self.feature_loss_coefficient * self.head_weights[idx+1]
        
        # Update metrics
        self.train_acc_heads[0](torch.argmax(outputs[0], dim=1), labels)
        for idx, output in enumerate(outputs[1:], 1):
            if idx < len(self.train_acc_heads):
                self.train_acc_heads[idx](torch.argmax(output, dim=1), labels)
        
        self.train_loss(loss)
        
        # Log metrics
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc_0", self.train_acc_heads[0], on_step=False, on_epoch=True)  # acc_0 for backbone
        
        # Log individual head accuracies with indices
        for idx in range(1, len(self.train_acc_heads)):
            self.log(f"train/acc_{idx}", self.train_acc_heads[idx], on_step=False, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step evaluating all heads."""
        # Get outputs based on configuration
        if self.use_all_heads:
            outputs, features, labels = self.model_step_all_heads(batch)
        else:
            outputs, features, labels = self.model_step_original(batch)
        
        labels = labels.to(self.device)
        
        total_loss = 0
        
        # Compute loss and accuracy for each head
        for i, output in enumerate(outputs):
            loss = self.criterion(output, labels)
            total_loss += loss
            
            if i < len(self.val_acc_heads):
                acc = self.val_acc_heads[i](torch.argmax(output, dim=1), labels)
                
                # Log individual metrics with indices (consistent with exp7_2)
                self.log(f"val/loss_{i}", loss, on_step=False, on_epoch=True)
                if i == 0:
                    # First output (backbone) with progress bar
                    self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True, prog_bar=True)
                else:
                    self.log(f"val/acc_{i}", acc, on_step=False, on_epoch=True)
        
        avg_loss = total_loss / len(outputs)
        self.val_loss(avg_loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Compute ensemble accuracy (weighted average of predictions)
        ensemble_logits = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = self.head_weights[i] if i < len(self.head_weights) else 1.0
            ensemble_logits += F.softmax(output, dim=1) * weight
        ensemble_preds = torch.argmax(ensemble_logits, dim=1)
        ensemble_acc = (ensemble_preds == labels).float().mean()
        self.log("val/acc_ensemble", ensemble_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"val_loss": avg_loss}

    def on_validation_epoch_end(self):
        """Reset validation metrics."""
        for acc_metric in self.val_acc_heads:
            acc_metric.reset()
        
        # Update best accuracy
        if len(self.val_acc_heads) > 0:
            acc = self.val_acc_heads[0].compute()
            self.val_acc_best.update(acc)
            self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Test step evaluating all heads."""
        # Get outputs based on configuration
        if self.use_all_heads:
            outputs, features, labels = self.model_step_all_heads(batch)
        else:
            outputs, features, labels = self.model_step_original(batch)
        
        labels = labels.to(self.device)
        
        # Evaluate backbone
        main_output = outputs[0]
        loss = self.criterion(main_output, labels)
        acc = self.test_acc_heads[0](torch.argmax(main_output, dim=1), labels)
        
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc_0", acc, on_step=False, on_epoch=True, prog_bar=True)  # acc_0 for backbone
        
        # Evaluate other heads with indices
        for idx, output in enumerate(outputs[1:], 1):
            if idx < len(self.test_acc_heads):
                acc = self.test_acc_heads[idx](torch.argmax(output, dim=1), labels)
                self.log(f"test/acc_{idx}", acc, on_step=False, on_epoch=True)
        
        # Compute ensemble accuracy
        ensemble_logits = torch.zeros_like(outputs[0])
        for i, output in enumerate(outputs):
            weight = self.head_weights[i] if i < len(self.head_weights) else 1.0
            ensemble_logits += F.softmax(output, dim=1) * weight
        ensemble_preds = torch.argmax(ensemble_logits, dim=1)
        ensemble_acc = (ensemble_preds == labels).float().mean()
        self.log("test/acc_ensemble", ensemble_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        """Reset test metrics."""
        pass

    def setup(self, stage: str) -> None:
        """Model initialization and setup."""
        if stage == "fit":
            pass  # Any specific setup for training

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = AnalogSGD(
            self.parameters(), 
            lr=self.hparams.optimizer['lr'],
            weight_decay=self.hparams.optimizer['weight_decay'],
            momentum=self.hparams.optimizer.get('momentum', 0.9),
            dampening=self.hparams.optimizer.get('dampening', 0),
            nesterov=self.hparams.optimizer.get('nesterov', False)
        )
        
        # Cosine Annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.scheduler.get('T_max', 300),
            eta_min=self.hparams.scheduler.get('eta_min', 0.0001)
        )
        
        scheduler_config = {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}
        
        return [optimizer], [scheduler_config]


if __name__ == "__main__":
    # Test instantiation
    _ = SalmonLitModule(
        model="test",
        integrated_resnet=None,
        compile=False,
        optimizer={'lr': 0.1, 'weight_decay': 0.0001},
        dataset="cifar10",
        epoch=100,
        loss_coefficient=0.5,
        feature_loss_coefficient=0.1,
        dataset_path="./data",
        autoaugment=False,
        temperature=3.0,
        batchsize=128,
        init_lr=0.1,
        N_CLASSES=10,
        block="BasicBlock",
        alpha=1.0,
        p_max=100,
        opt_config="sgd",
        sd_config="standard",
        FC_Digit="digital",
        sch_config="cosine",
        scheduler={'T_max': 300, 'eta_min': 0.0001}
    )
    print("Module initialized successfully!")