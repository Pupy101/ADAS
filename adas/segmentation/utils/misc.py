from collections import OrderedDict
from typing import Any, List, Mapping, Tuple, Type, Union

from catalyst import dl
from catalyst.core.callback import Callback
from catalyst.utils import load_checkpoint
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from adas.core.utils.misc import create_callbacks as create_base_callbacks

from ..config import CLASS_NAMES, Config
from ..models.types import ModelType
from ..models.u2net import U2net
from ..models.unet import Unet


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
    callbacks = create_base_callbacks(config=config)
    segmentation_callbacks = [
        dl.CriterionCallback(input_key="probas", target_key="targets", metric_key="loss"),
        dl.IOUCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
        dl.DiceCallback(input_key="last_probas", target_key="targets", class_names=CLASS_NAMES),
    ]
    callbacks.extend(segmentation_callbacks)

    return callbacks


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
