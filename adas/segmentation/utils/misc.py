from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.loggers.wandb import WandbLogger
from catalyst.utils import load_checkpoint
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from adas.segmentation.config import CLASS_NAMES, Config, TrainCfg
from adas.segmentation.models.types import ModelType
from adas.segmentation.models.u2net import U2net
from adas.segmentation.models.unet import Unet


class SegmentationRunner(dl.SupervisedRunner):  # pylint: disable=missing-class-docstring
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        outputs = self.forward(batch)
        logits: Tuple[Tensor, ...] = outputs["logits"]
        outputs["probas"] = tuple(F.softmax(l, dim=1) for l in logits)
        outputs["last_probas"] = outputs["probas"][-1]
        self.batch = {**batch, **outputs}


def create_model(config: Config) -> Union[Unet, U2net]:
    """Create model from config"""

    model_type: Union[Type[Unet], Type[U2net]]
    if config.model is ModelType.U2NET:
        model_type = U2net
    elif config.model is ModelType.UNET:
        model_type = Unet
    else:
        raise TypeError(f"Strange config type: {config}")
    return model_type(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        size=config.size.value,
        downsample_mode=config.downsample.value,
        upsample_mode=config.upsample.value,
        count_features=config.count_features,
    )


def create_callbacks(config: Config) -> List[Callback]:
    """Create callback for catalyst runner"""
    callbacks: List[Callback] = [
        dl.CheckpointCallback(
            logdir=config.logdir,
            loader_key="valid",
            metric_key="loss",
            topk=3,
            resume_model=config.resume,
            mode="runner",
        ),
        dl.EarlyStoppingCallback(loader_key="valid", metric_key="loss", minimize=True, patience=3),
        dl.CriterionCallback(input_key="probas", target_key="targets", metric_key="loss"),
        dl.IOUCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
        dl.DiceCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
    ]
    if isinstance(config, TrainCfg):
        if config.num_batch_steps is not None:
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

    return callbacks


def create_logger(config: Config) -> Optional[Dict[str, Any]]:
    """Create wandb logger or return None"""
    if config.wandb and config.name_run is None:
        raise ValueError("Please set parameter --name_run for logging to wandb")
    if config.wandb:
        return {"wandb": WandbLogger(project="ADAS", name=config.name_run, log_batch_metrics=True)}
    return None


def load_encoder_weights(
    weight: str,
    model: Module,
    encoder_name: str = "feature_extractor",
    mode: str = "runner",
) -> Module:
    """Load encoder weights from trained classificator"""
    assert mode in ["runner", "model"]
    weights = load_checkpoint(weight)
    model_weight: OrderedDict = weight if mode == "model" else weights["model_state_dict"]
    encoder_weights = OrderedDict()
    for k, v in model_weight.items():
        if not k.startswith(encoder_name + "."):
            continue
        new_k = k.replace(encoder_name + ".", "", 1)
        encoder_weights[new_k] = v
    model.encoder.load_state_dict(encoder_weights)  # type: ignore
    return model
