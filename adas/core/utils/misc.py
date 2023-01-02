from typing import Any, Dict, List, Optional, TypeVar

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.loggers import WandbLogger

from ..config import HyperParams, LoggingParams, WeightLoggingParams

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
