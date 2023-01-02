from typing import List, Type, Union

from catalyst import dl
from catalyst.core.callback import Callback

from adas.core.utils.misc import create_callbacks as create_base_callbacks

from ..config import Config
from ..models.segmentation import U2netEncoderClassifier, UnetEncoderClassifier
from ..models.types import ModelType


def create_model(config: Config) -> Union[U2netEncoderClassifier, UnetEncoderClassifier]:
    """Create model from config"""

    model_type: Union[Type[U2netEncoderClassifier], Type[UnetEncoderClassifier]]
    if config.model is ModelType.U2NET_ENCODER:
        model_type = U2netEncoderClassifier
    elif config.model is ModelType.UNET_ENCODER:
        model_type = UnetEncoderClassifier
    else:
        raise TypeError(f"Strange config type: {config}")
    return model_type(
        in_channels=config.in_channels,
        size=config.size.value,
        downsample_mode=config.downsample.value,
        count_classes=config.classes,
    )


def create_callbacks(config: Config) -> List[Callback]:
    """Create callback for catalyst runner"""
    callbacks = create_base_callbacks(config=config)
    classification_callbacks = [
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=[1, 5]),
        dl.AUCCallback(input_key="logits", target_key="targets"),
        dl.PrecisionRecallF1SupportCallback(
            input_key="logits", target_key="targets", num_classes=config.classes
        ),
    ]
    callbacks.extend(classification_callbacks)

    return callbacks
