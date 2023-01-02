import random
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.loggers import WandbLogger

from ..config import HyperParams, LoggingParams, WeightLoggingParams

T = TypeVar("T")
EnumType = TypeVar("EnumType", bound=Enum)  # pylint: disable=invalid-name
CallbackConfig = TypeVar("CallbackConfig", HyperParams, WeightLoggingParams)
LoggerConfig = TypeVar("LoggerConfig", bound=LoggingParams)


def create_callbacks(config: CallbackConfig) -> List[Callback]:
    """Create callback for catalyst runner"""
    callbacks: List[Callback] = [
        dl.EarlyStoppingCallback(loader_key="valid", metric_key="loss", minimize=True, patience=3),
    ]
    if isinstance(config, HyperParams):
        if config.num_batch_steps:
            callbacks.append(
                dl.CheckRunCallback(
                    num_batch_steps=config.num_batch_steps,
                    num_epoch_steps=config.num_epochs,
                )
            )
        if config.profile:
            callbacks.append(
                dl.ProfilerCallback(
                    loader_key="train",
                    num_batches=10,
                    profiler_kwargs={
                        "record_shapes": True,
                        "profile_memory": True,
                        "with_flops": True,
                        "with_modules": True,
                    },
                )
            )
    if isinstance(config, WeightLoggingParams):
        callbacks.append(
            dl.CheckpointCallback(
                logdir=config.logdir,
                loader_key="valid",
                metric_key="loss",
                topk=3,
                resume_model=config.resume,
                mode="runner",
            )
        )
    return callbacks


def create_logger(config: LoggerConfig) -> Optional[Dict[str, Any]]:
    """Create wandb logger or return None"""
    if config.wandb and config.name_run is None:
        raise ValueError("Please set parameter --name_run for logging to wandb")
    if config.wandb:
        return {"wandb": WandbLogger(project="ADAS", name=config.name_run, log_batch_metrics=True)}
    return None


def train_test_split(data: List[T], test_size: float, seed: int) -> Tuple[List[T], List[T]]:
    """Split sequence of items on train/val subsequences"""
    random.seed(seed)
    assert 0 < test_size < 1, "test_size must in interval (0, 1)"
    train_data: List[T] = []
    test_data: List[T] = []
    for item in data:
        if random.random() < test_size:
            test_data.append(item)
        else:
            train_data.append(item)
    return train_data, test_data


def find_enum(value: Any, enum_type: Type[EnumType]) -> EnumType:
    """Find enum by it value or raise ValueError"""
    for elem in enum_type:
        if value == elem.value:
            return elem
    raise ValueError(f"Strange value {value} for enum {enum_type}")
