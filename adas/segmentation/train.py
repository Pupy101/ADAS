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
from adas.segmentation.utils.common import parse_train_args
from adas.segmentation.utils.configs import CLASS_NAMES, DatasetArgs, DDPConfig, ModelType


class DDPRunner(dl.SupervisedRunner):
    """Distributed catalyst runner."""

    def __init__(  # pylint: disable=too-many-arguments
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
                train_data,
                sampler=DistributedSampler(dataset=train_data),
                batch_size=self.config.train_batch_size,
            ),
            "valid": DataLoader(
                valid_data,
                sampler=DistributedSampler(dataset=valid_data),
                batch_size=self.config.valid_batch_size,
            ),
        }
        return loaders


class MultipleOutputModelRunner(dl.SupervisedRunner):
    """Multi output model catalyst runner."""

    def __init__(  # pylint: disable=too-many-arguments
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
        self.batch = {
            **batch,
            "probas": probas,
            "last_probas": probas[-2],
            "agg_probas": probas[-1],
        }


class DDPMultipleOutputModelRunner(  # pylint: disable=too-many-ancestors
    DDPRunner, MultipleOutputModelRunner
):
    """Distributed catalyst runner for multi output model."""

    pass  # pylint: disable=unnecessary-pass


if __name__ == "__main__":
    TRAIN_CONFIG, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE = parse_train_args()

    SEGMENTATION_MODEL: Union[Unet, U2net]

    if TRAIN_CONFIG.model is ModelType.UNET:
        SEGMENTATION_MODEL = Unet(
            in_channels=TRAIN_CONFIG.in_channels,
            out_channels=TRAIN_CONFIG.out_channels,
            big=TRAIN_CONFIG.big,
            max_pool=TRAIN_CONFIG.max_pool,
            bilinear=TRAIN_CONFIG.bilinear,
        )
    else:
        SEGMENTATION_MODEL = U2net(
            in_channels=TRAIN_CONFIG.in_channels,
            out_channels=TRAIN_CONFIG.out_channels,
            big=TRAIN_CONFIG.big,
            max_pool=TRAIN_CONFIG.max_pool,
            bilinear=TRAIN_CONFIG.bilinear,
        )

    OPTIMIZER = AdamW(SEGMENTATION_MODEL.parameters(), lr=TRAIN_CONFIG.learning_rate)

    CRITERION = AggregatorManyOutputsLoss(
        losses=DiceLoss(class_dim=TRAIN_CONFIG.out_channels),
        coefficients=(0.01, 0.05, 0.2, 0.5, 1, 0.5),
    )

    LOGGER = (
        {"wandb": WandbLogger(project="ADAS", name="Unet_test_run", log_batch_metrics=True)}
        if TRAIN_CONFIG.logging
        else None
    )

    TRAIN_DATASET_ARGS = DatasetArgs(
        image_dir="/content/train/images",
        mask_dir="/content/train/roads",
        transforms=create_train_augmentation(),
    )
    VALID_DATASET_ARGS = DatasetArgs(
        image_dir="/content/val/images",
        mask_dir="/content/val/roads",
        transforms=create_train_augmentation(is_train=False),
    )

    CALLBACKS = [
        dl.IOUCallback(
            input_key="last_probas",
            target_key="targets",
            prefix="last_",
            class_names=CLASS_NAMES,
        ),
        dl.DiceCallback(
            input_key="last_probas",
            target_key="targets",
            prefix="last_",
            class_names=CLASS_NAMES,
        ),
        dl.IOUCallback(
            input_key="agg_probas",
            target_key="targets",
            prefix="agg_",
            class_names=CLASS_NAMES,
        ),
        dl.DiceCallback(
            input_key="agg_probas",
            target_key="targets",
            prefix="agg_",
            class_names=CLASS_NAMES,
        ),
        dl.CheckpointCallback(
            logdir=TRAIN_CONFIG.logdir,
            loader_key="valid",
            metric_key="last_iou",
            topk=3,
        ),
        dl.EarlyStoppingCallback(
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            patience=3,
            min_delta=1e-2,
        ),
        dl.ProfilerCallback(loader_key="valid"),
    ]

    TRAIN_DDP_CONFIG = DDPConfig(
        datasets={"train": TRAIN_DATASET_ARGS, "valid": VALID_DATASET_ARGS},
        train_batch_size=TRAIN_BATCH_SIZE,
        valid_batch_size=VALID_BATCH_SIZE,
    )

    if TRAIN_CONFIG.ddp:
        RUNNER = DDPMultipleOutputModelRunner(TRAIN_DDP_CONFIG)
    else:
        TRAIN_DATASET = BDD100KDataset(**TRAIN_DATASET_ARGS.asdict())
        VALID_DATASET = BDD100KDataset(**VALID_DATASET_ARGS.asdict())
        TRAIN_CONFIG.loaders = {
            "train": DataLoader(TRAIN_DATASET, batch_size=TRAIN_BATCH_SIZE, shuffle=True),
            "valid": DataLoader(VALID_DATASET, batch_size=VALID_BATCH_SIZE, shuffle=False),
        }
        RUNNER = MultipleOutputModelRunner()

    HPARAMS = TRAIN_CONFIG.asdict(exclude=["model", "loaders"])
    HPARAMS["model"] = TRAIN_CONFIG.model.value
    if isinstance(CRITERION, AggregatorManyOutputsLoss):
        HPARAMS["loss_coefs"] = CRITERION.coeffs  # save coefficients
    ADDITIONAL_PARAMS_RUNNER = TRAIN_CONFIG.asdict(
        exclude=[
            "model",
            "in_channels",
            "out_channels",
            "big",
            "max_pool",
            "bilinear",
            "learning_rate",
        ]
    )

    RUNNER.train(
        model=SEGMENTATION_MODEL,
        OPTIMIZER=OPTIMIZER,
        CRITERION=CRITERION,
        loggers=LOGGER,
        CALLBACKS=CALLBACKS,
        hparams=HPARAMS,
        **ADDITIONAL_PARAMS_RUNNER
    )
