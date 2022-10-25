from typing import Any, Mapping, Optional, Union

from catalyst import dl
from catalyst.contrib.losses import DiceLoss
from catalyst.core.engine import Engine
from catalyst.core.runner import RunnerModel
from catalyst.loggers.wandb import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from adas.common.loss import AggregatorManyOutputsLoss
from adas.segmentation.data import BDD100KDataset, create_train_augmentation
from adas.segmentation.models import U2net, Unet
from adas.segmentation.utils.configs import CLASS_NAMES, DatasetArgs, DDPConfig, ModelType, TrainConfig


class DDPRunner(dl.SupervisedRunner):
    def __init__(
        self,
        config: DDPConfig,
        model: RunnerModel = None,
        engine: Optional[Engine] = None,
        input_key: str = "features",
        output_key: str = "probas",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        self.config = config
        super().__init__(model, engine, input_key, output_key, target_key, loss_key)

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
    def __init__(
        self,
        model: RunnerModel = None,
        engine: Engine = None,
        input_key: Any = "features",
        output_key: Any = "probas",
        target_key: str = "targets",
        loss_key: str = "loss",
    ):
        super().__init__(model, engine, input_key, output_key, target_key, loss_key)

    def handle_batch(self, batch: Mapping[str, Any]):
        probas = self.forward(batch)["probas"]
        self.batch = {**batch, "probas": probas, "last_probas": probas[-2], "agg_probas": probas[-1]}


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

segmentation_model: Union[Unet, U2net]

if config.model is ModelType.UNET:
    segmentation_model = Unet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        big=config.big,
        max_pool=config.max_pool,
        bilinear=config.bilinear,
    )
else:
    segmentation_model = U2net(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        big=config.big,
        max_pool=config.max_pool,
        bilinear=config.bilinear,
    )

optimizer = AdamW(segmentation_model.parameters(), lr=1e-3)

scheduler = None

criterion = AggregatorManyOutputsLoss(
    losses=DiceLoss(class_dim=config.out_channels),
    coefficients=(0.01, 0.02, 0.1, 0.2, 1, 0.7),
)


logger = {"wandb": WandbLogger(project="ADAS", name="Unet", log_batch_metrics=True)}

train_dataset_args = DatasetArgs(
    image_dir="/content/train/images", mask_dir="/content/train/roads", transforms=create_train_augmentation()
)
valid_dataset_args = DatasetArgs(
    image_dir="/content/val/images", mask_dir="/content/val/roads", transforms=create_train_augmentation(is_train=False)
)

callbacks = [
    dl.IOUCallback(input_key="last_probas", target_key="targets", prefix="last_", class_names=CLASS_NAMES),
    dl.DiceCallback(input_key="last_probas", target_key="targets", prefix="last_", class_names=CLASS_NAMES),
    dl.IOUCallback(input_key="agg_probas", target_key="targets", prefix="agg_", class_names=CLASS_NAMES),
    dl.DiceCallback(input_key="agg_probas", target_key="targets", prefix="agg_", class_names=CLASS_NAMES),
    dl.CheckpointCallback(logdir=config.logdir, loader_key="valid", metric_key="last_iou", topk=3),
]

train_ddp_config = DDPConfig(
    datasets={"train": train_dataset_args, "valid": valid_dataset_args},
    train_batch_size=train_batch_size,
    valid_batch_size=valid_batch_size,
)

if config.ddp:
    runner = DDPMultipleOutputModelRunner(train_ddp_config)
else:
    train_dataset = BDD100KDataset(**train_dataset_args.asdict())
    valid_dataset = BDD100KDataset(**valid_dataset_args.asdict())
    config.loaders = {
        "train": DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True),
        "valid": DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False),
    }
    runner = MultipleOutputModelRunner()

runner.train(
    model=segmentation_model,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    loggers=logger,
    callbacks=callbacks,
    hparams=config.asdict(exclude=["model", "loaders"]),
    **config.asdict(exclude=["model", "in_channels", "out_channels", "big", "max_pool", "bilinear"])
)
