from typing import Any, Dict, List, Mapping, Optional, Union

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.core.engine import Engine
from catalyst.core.runner import RunnerModel
from catalyst.loggers.wandb import WandbLogger

from adas.segmentation.models import U2net, Unet
from adas.segmentation.utils.configs import (
    CLASS_NAMES,
    EvaluationConfig,
    InferenceConfig,
    ModelType,
    TrainConfig,
)


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
        }


def create_callbacks(  # pylint: disable=too-many-arguments
    logdir: str,
    resume: Optional[str],
    profile: bool,
    num_batch_steps: Optional[int] = None,
    num_epoch_steps: Optional[int] = None,
) -> List[Callback]:
    """Create callback for catalyst runner"""
    callbacks: List[Callback]
    callbacks = [
        dl.CheckpointCallback(
            logdir=logdir, loader_key="valid", metric_key="loss", topk=3, resume_model=resume
        ),
        dl.EarlyStoppingCallback(
            loader_key="valid", metric_key="loss", minimize=True, patience=3, min_delta=1e-2
        ),
        dl.IOUCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
        dl.DiceCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
    ]
    if profile:
        callbacks.append(dl.ProfilerCallback(loader_key="valid"))
    if num_batch_steps is not None and num_epoch_steps is not None:
        callbacks.append(
            dl.CheckRunCallback(num_batch_steps=num_batch_steps, num_epoch_steps=num_epoch_steps)
        )
    return callbacks


def create_model(
    config: Union[EvaluationConfig, TrainConfig, InferenceConfig]
) -> Union[U2net, Unet]:
    """Create model from config"""
    kwargs: Mapping[str, Any] = {
        "in_channels": config.in_channels,
        "out_channels": config.out_channels,
        "big": config.big,
        "max_pool": config.max_pool,
        "bilinear": config.bilinear,
    }
    return Unet(**kwargs) if config.model is ModelType.UNET else U2net(**kwargs)


def create_logger(config: Union[EvaluationConfig, TrainConfig]) -> Optional[Dict[str, Any]]:
    """Create wandb logger or return None"""
    if config.logging:
        return {"wandb": WandbLogger(project="ADAS", name=config.name, log_batch_metrics=True)}
    return None
