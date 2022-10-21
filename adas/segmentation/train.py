from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from catalyst import dl
from catalyst.callbacks import Callback
from catalyst.contrib.losses import FocalLossMultiClass
from catalyst.core.engine import Engine
from catalyst.core.logger import ILogger
from catalyst.loggers.wandb import WandbLogger
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from adas.common.loss import AggregatorManyOutputsLoss
from adas.segmentation.data import BDD100KDataset, create_train_augmentation
from adas.segmentation.models import Unet
from adas.segmentation.utils.configs import DatasetArgs, TrainDDPConfig


class DDPRunner(dl.SupervisedRunner):
    def __init__(
        self,
        model: Optional[Module] = None,
        engine: Optional[Engine] = None,
        input_key: str = "features",
        output_key: str = "logits",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        self.config: TrainDDPConfig
        super().__init__(model, engine, input_key, output_key, target_key, loss_key)

    def train(
        self,
        *,
        config: TrainDDPConfig,
        loaders: Mapping[str, DataLoader],
        model: Module,
        criterion: Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        engine: Optional[Union[Engine, str]] = None,
        callbacks: Optional[Union[List[Callback], Mapping[str, Callback]]] = None,
        loggers: Dict[str, ILogger] = None,
        seed: int = 42,
        hparams: Dict[str, Any] = None,
        num_epochs: int = 1,
        logdir: str = None,
        resume: str = None,
        valid_loader: str = None,
        valid_metric: str = None,
        minimize_valid_metric: bool = None,
        verbose: bool = False,
        timeit: bool = False,
        check: bool = False,
        overfit: bool = False,
        profile: bool = False,
        load_best_on_end: bool = False,
        cpu: bool = False,
        fp16: bool = False,
        ddp: bool = False
    ) -> None:
        self.config = config
        return super().train(
            loaders=loaders,
            model=model,
            engine=engine,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=callbacks,
            loggers=loggers,
            seed=seed,
            hparams=hparams,
            num_epochs=num_epochs,
            logdir=logdir,
            resume=resume,
            valid_loader=valid_loader,
            valid_metric=valid_metric,
            minimize_valid_metric=minimize_valid_metric,
            verbose=verbose,
            timeit=timeit,
            check=check,
            overfit=overfit,
            profile=profile,
            load_best_on_end=load_best_on_end,
            cpu=cpu,
            fp16=fp16,
            ddp=ddp,
        )

    def get_engine(self) -> Engine:
        return dl.DistributedDataParallelEngine()

    def get_loaders(self) -> Mapping[str, DataLoader]:
        assert self.config.datasets is not None
        train_data = BDD100KDataset(**self.config.datasets["train"].asdict())
        valid_data = BDD100KDataset(**self.config.datasets["train"].asdict())
        loaders = {
            "train": DataLoader(
                train_data, sampler=DistributedSampler(dataset=train_data), batch_size=self.config.train_batch_size
            ),
            "valid": DataLoader(
                valid_data, sampler=DistributedSampler(dataset=valid_data), batch_size=self.config.valid_batch_size
            ),
        }
        return loaders


class MultipleOutputModelRunner(dl.SupervisedRunner):
    def handle_batch(self, batch: Mapping[str, Any]):
        logits = self.forward(batch)["logits"]
        probas = torch.softmax(logits[-1], dim=1)
        self.batch = {**batch, "logits": logits, "overall_logits": logits[-1], "probas": probas}


class DDPMultipleOutputModelRunner(DDPRunner, MultipleOutputModelRunner):
    pass


train_kwargs = {
    "seed": 123456,
    "num_epochs": 1,
    "logdir": "./logs",
    "resume": None,
    "valid_loader": "valid",
    "valid_metric": "loss",
    "verbose": True,
    "cpu": True,
    "fp16": True,
    "ddp": False,
    "minimize_valid_metric": True,
}

train_batch_size = 2
valid_batch_size = 2

model = Unet(in_channels=3, out_channels=3)

optimizer = AdamW(model.parameters(), lr=1e-4)

scheduler = None

criterion = AggregatorManyOutputsLoss(losses=FocalLossMultiClass(), coefficients=(0.1, 0.2, 0.3, 0.4, 0.5, 1))

logger = {"wandb": WandbLogger(project="ADAS", name="Unet", log_batch_metrics=True)}

train_dataset_args = DatasetArgs(
    image_dir="/Users/19891176/Downloads/dataset/train/images",
    mask_dir="/Users/19891176/Downloads/dataset/train/roads",
    transforms=create_train_augmentation(),
)
valid_dataset_args = DatasetArgs(
    image_dir="/Users/19891176/Downloads/dataset/val/images",
    mask_dir="/Users/19891176/Downloads/dataset/val/roads",
    transforms=create_train_augmentation(is_train=False),
)

callbacks = [
    dl.IOUCallback(input_key="probas", target_key="masks"),
    dl.DiceCallback(input_key="probas", target_key="masks"),
]

train_ddp_config = TrainDDPConfig(
    datasets={"train": train_dataset_args, "valid": valid_dataset_args},
    train_batch_size=train_batch_size,
    valid_batch_size=valid_batch_size,
)

if train_kwargs["ddp"]:
    runner = DDPMultipleOutputModelRunner()
    train_kwargs["config"] = train_ddp_config
else:
    train_dataset = BDD100KDataset(**train_dataset_args.asdict())
    valid_dataset = BDD100KDataset(**valid_dataset_args.asdict())
    train_kwargs["loaders"] = {
        "train": DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
        "valid": DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False),
    }
    runner = MultipleOutputModelRunner()

runner.train(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    loggers=logger,
    callbacks=callbacks,
    **train_kwargs  # type: ignore
)
