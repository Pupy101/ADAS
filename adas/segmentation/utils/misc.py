from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Optional, Tuple

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.loggers.wandb import WandbLogger
from catalyst.utils import load_checkpoint
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from adas.segmentation.configs import (
    CLASS_NAMES,
    Config,
    EvaluationEncoderConfig,
    EvaluationSegmentationConfig,
    TrainEncoderConfig,
    TrainSegmentationConfig,
)
from adas.segmentation.models import Classificator, ModelType, U2net, Unet
from adas.segmentation.models.u2net import U2netEncoder
from adas.segmentation.models.unet import UnetEncoder


class SegmentationRunner(dl.SupervisedRunner):  # pylint: disable=missing-class-docstring
    def handle_batch(self, batch: Mapping[str, Any]) -> None:
        outputs = self.forward(batch)
        logits: Tuple[Tensor, ...] = outputs["logits"]
        outputs["probas"] = tuple(F.softmax(l, dim=1) for l in logits)
        outputs["last_probas"] = outputs["probas"][-1]
        self.batch = {**batch, **outputs}


def create_model(config: Config) -> Module:
    """Create model from config"""
    kwargs: Dict[str, Any] = {
        "in_channels": config.in_channels,
        "size": config.size,
        "max_pool": config.max_pool,
    }
    if isinstance(config, (TrainSegmentationConfig, EvaluationSegmentationConfig)):
        kwargs.update(
            {
                "out_channels": config.out_channels,
                "bilinear": config.bilinear,
                "count_predict_masks": config.count_predict_masks,
            }
        )
        model = (Unet if config.model is ModelType.UNET else U2net)(**kwargs)
        return model
    if isinstance(config, (TrainEncoderConfig, EvaluationEncoderConfig)):
        extractor = (UnetEncoder if config.model is ModelType.UNET else U2netEncoder)(**kwargs)
        return Classificator(
            extractor,  # type: ignore
            count_classes=config.count_classes,
            dropout=config.dropout,
        )
    raise TypeError(f"Strange config type: {config}")


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
    ]
    if isinstance(config, (TrainSegmentationConfig, EvaluationSegmentationConfig)):
        metric_callbacks = [
            dl.CriterionCallback(input_key="probas", target_key="targets", metric_key="loss"),
            dl.IOUCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
            dl.DiceCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
        ]
        callbacks.extend(metric_callbacks)
    elif isinstance(config, (TrainEncoderConfig, EvaluationEncoderConfig)):
        metric_callbacks = [
            dl.AccuracyCallback(
                input_key="logits",
                target_key="targets",
                num_classes=config.count_classes,
                topk=[1, 5],
            ),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits",
                target_key="targets",
                num_classes=config.count_classes,
                zero_division=1,
            ),
        ]
        callbacks.extend(metric_callbacks)
    if isinstance(config, (TrainSegmentationConfig, TrainEncoderConfig)):
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
    if config.logging:
        return {"wandb": WandbLogger(project="ADAS", name=config.name_run, log_batch_metrics=True)}
    return None


def load_encoder_weights(
    weight: str, model: Module, encoder_name: str = "feature_extractor", mode: str = "runner"
) -> Module:
    assert mode in ["runner", "model"]
    weights = load_checkpoint(weight)
    model_weight: OrderedDict = weight if mode == "model" else weights["model_state_dict"]
    encoder_weights = OrderedDict()
    for k, v in model_weight.items():
        if not k.startswith(encoder_name + "."):
            continue
        new_k = k.replace(encoder_name + ".", "", 1)
        encoder_weights[new_k] = v
    model.encoder.load_state_dict(encoder_weights)
    return model
