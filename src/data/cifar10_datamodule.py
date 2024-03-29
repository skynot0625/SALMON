from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import autoaugment

class Cifar10DataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (45_000, 5_000, 10_000),
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = False,
        autoaugment: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.mean = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.Tensor([0.2023, 0.1994, 0.2010])

        self.autoaugment = autoaugment
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
        datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        if not self.data_train and not self.data_val and not self.data_test:
            transform_train = self._create_train_transform()
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ])

            trainset = datasets.CIFAR10(self.hparams.data_dir, train=True, transform=transform_train)
            testset = datasets.CIFAR10(self.hparams.data_dir, train=False, transform=transform_test)
            dataset = ConcatDataset([trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def _create_train_transform(self):
        transforms_list = [
            transforms.RandomCrop(32, padding=4, fill=128),
            transforms.RandomHorizontalFlip()
        ]
        if self.autoaugment:
            transforms_list.append(autoaugment.CIFAR10Policy())

        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        return transforms.Compose(transforms_list)

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size_per_device,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            worker_init_fn=lambda worker_id: torch.manual_seed(42 + worker_id)
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

if __name__ == "__main__":
    _ = Cifar10DataModule()
