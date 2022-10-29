from typing import Any, Mapping, Optional, Union

from catalyst import dl
from catalyst.contrib.losses import DiceLoss
from catalyst.core.engine import Engine
from catalyst.core.runner import RunnerModel
from catalyst.loggers.wandb import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler

from adas.common.loss import AggregatorManyOutputsLoss
from adas.common.utils import train_test_split
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
        train_valid_data = BDD100KDataset(**self.config.dataset.asdict())
        train_data, valid_data = train_test_split(
            train_valid_data, test_size=self.config.valid_size, seed=self.config.seed
        )
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
    TRAIN_CONFIG = parse_train_args()

    SEGMENTATION_MODEL: Union[Unet, U2net]

    MODEL_ARGS: Mapping[str, Any] = {
        "in_channels": TRAIN_CONFIG.in_channels,
        "out_channels": TRAIN_CONFIG.out_channels,
        "big": TRAIN_CONFIG.big,
        "max_pool": TRAIN_CONFIG.max_pool,
        "bilinear": TRAIN_CONFIG.bilinear,
    }

    if TRAIN_CONFIG.model is ModelType.UNET:
        SEGMENTATION_MODEL = Unet(**MODEL_ARGS)
    else:
        SEGMENTATION_MODEL = U2net(**MODEL_ARGS)

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

    TRAIN_VALID_DATASET_ARGS = DatasetArgs(
        image_dir="/content/train/images",
        mask_dir="/content/train/roads",
        transforms=create_train_augmentation(),
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
    if TRAIN_CONFIG.count_batches is not None:
        CALLBACKS.append(
            dl.CheckRunCallback(
                num_batch_steps=TRAIN_CONFIG.count_batches,
                num_epoch_steps=TRAIN_CONFIG.num_epochs,
            )
        )

    TRAIN_DDP_CONFIG = DDPConfig(
        dataset=TRAIN_VALID_DATASET_ARGS,
        train_batch_size=TRAIN_CONFIG.train_batch_size,
        valid_batch_size=TRAIN_CONFIG.valid_batch_size,
        valid_size=TRAIN_CONFIG.valid_size,
        seed=TRAIN_CONFIG.seed,
    )

    if TRAIN_CONFIG.ddp:
        RUNNER = DDPMultipleOutputModelRunner(TRAIN_DDP_CONFIG)
    else:
        TRAIN_VALID_DATASET = BDD100KDataset(**TRAIN_VALID_DATASET_ARGS.asdict())
        TRAIN_DATASET, VALID_DATASET = train_test_split(
            TRAIN_VALID_DATASET, test_size=0.3, seed=TRAIN_CONFIG.seed
        )
        TRAIN_CONFIG.loaders = {
            "train": DataLoader(
                TRAIN_DATASET, batch_size=TRAIN_CONFIG.train_batch_size, shuffle=True
            ),
            "valid": DataLoader(
                VALID_DATASET, batch_size=TRAIN_CONFIG.valid_batch_size, shuffle=False
            ),
        }
        RUNNER = MultipleOutputModelRunner()

    HPARAMS = TRAIN_CONFIG.asdict(exclude=["model", "loaders"])
    HPARAMS["model"] = TRAIN_CONFIG.model.value
    if isinstance(CRITERION, AggregatorManyOutputsLoss):
        HPARAMS["loss_coefs"] = CRITERION.coeffs  # save coefficients
    ADDITIONAL_PARAMS_RUNNER = TRAIN_CONFIG.asdict(
        exclude=[
            "big",
            "bilinear",
            "count_batches",
            "in_channels",
            "learning_rate",
            "logging",
            "max_pool",
            "model",
            "out_channels",
            "train_batch_size",
            "valid_batch_size",
            "valid_size",
        ]
    )

    RUNNER.train(
        model=SEGMENTATION_MODEL,
        optimizer=OPTIMIZER,
        criterion=CRITERION,
        loggers=LOGGER,
        callbacks=CALLBACKS,
        hparams=HPARAMS,
        **ADDITIONAL_PARAMS_RUNNER
    )
