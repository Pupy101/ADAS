from typing import Any, Dict, List, Mapping, Optional, Union

import torch
from catalyst import dl
from catalyst.callbacks import Callback
from catalyst.contrib.losses import DiceLoss, FocalLossMultiClass
from catalyst.core.engine import Engine
from catalyst.core.logger import ILogger
from catalyst.loggers.wandb import WandbLogger
from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, DistributedSampler

from adas.common.loss import AggregatorManyOutputsLoss
from adas.segmentation.data import BDD100KDataset, create_train_augmentation
from adas.segmentation.models import U2net, Unet
from adas.segmentation.utils.configs import DatasetArgs, DDPConfig, ModelType, TrainConfig


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
        self.config: DDPConfig
        super().__init__(model, engine, input_key, output_key, target_key, loss_key)

    def train(
        self,
        *,
        config: DDPConfig,
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


config = TrainConfig(
    model=ModelType.UNET,
    in_channels=3,
    out_channels=3,
    big=False,
    max_pool=True,
    bilinear=True,
    seed=123456,
    num_epochs=20,
    logdir="./logs",
    cpu=False,
)

train_batch_size = 40
valid_batch_size = 80

model: Union[Unet, U2net]

if config.model is ModelType.UNET:
    model = Unet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        big=config.big,
        max_pool=config.max_pool,
        bilinear=config.bilinear,
    )
else:
    model = U2net(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        big=config.big,
        max_pool=config.max_pool,
        bilinear=config.bilinear,
    )

optimizer = AdamW(model.parameters(), lr=1e-4)

scheduler = None

criterion = {
    "focal": AggregatorManyOutputsLoss(losses=FocalLossMultiClass(), coefficients=(0.1, 0.2, 0.3, 0.4, 0.5, 1)),
    "dice": AggregatorManyOutputsLoss(
        losses=DiceLoss(class_dim=config.out_channels), coefficients=(0.1, 0.2, 0.3, 0.4, 0.5, 1)
    ),
}


logger = {"wandb": WandbLogger(project="ADAS", name="Unet", log_batch_metrics=True)}

train_dataset_args = DatasetArgs(
    image_dir="/content/train/images", mask_dir="/content/train/roads", transforms=create_train_augmentation()
)
valid_dataset_args = DatasetArgs(
    image_dir="/content/val/images", mask_dir="/content/val/roads", transforms=create_train_augmentation(is_train=False)
)

callbacks = [
    dl.IOUCallback(input_key="probas", target_key="masks"),
    dl.DiceCallback(input_key="probas", target_key="masks"),
    dl.CriterionCallback(input_key="logits", target_key="targets", metric_key="loss_focal", criterion_key="focal"),
    dl.CriterionCallback(input_key="probas", target_key="masks", metric_key="loss_dice", criterion_key="dice"),
    dl.MetricAggregationCallback(metric_key="loss", metrics={"loss_focal": 0.4, "loss_dice": 0.6}, mode="weighted_sum"),
    dl.BackwardCallback(metric_key="loss", log_gradient=True),
]

train_ddp_config = DDPConfig(
    datasets={"train": train_dataset_args, "valid": valid_dataset_args},
    train_batch_size=train_batch_size,
    valid_batch_size=valid_batch_size,
)

if config.ddp:
    runner = DDPMultipleOutputModelRunner()
    config.ddp_config = train_ddp_config
else:
    train_dataset = BDD100KDataset(**train_dataset_args.asdict())
    valid_dataset = BDD100KDataset(**valid_dataset_args.asdict())
    config.loaders = {
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
    hparams=config.asdict(exclude=["model", "ddp_config", "loaders"]),
    **config.asdict(exclude=["model", "in_channels", "out_channels", "big", "max_pool", "bilinear", "ddp_config"])
)
