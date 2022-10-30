from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

CLASS_NAMES = ["main_road", "backgroud"]


@dataclass
class AsDictDataclass:
    """Mixin class for factory method 'asdict'"""

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Method for representation dataclass as dict"""
        dictionary = asdict(self)
        if exclude is not None:
            for key in exclude:
                dictionary.pop(key)
        return dictionary


class ModelType(Enum):
    """Acceptable models types"""

    UNET = "UNET"
    U2NET = "U2NET"


COEFFICIENTS = {
    ModelType.UNET: (0.01, 0.05, 0.1, 0.4, 1),
    ModelType.U2NET: (0.001, 0.01, 0.05, 0.1, 0.4, 1),
}


@dataclass
class ModelParameters:
    """Base segmentation model parameters"""

    model: ModelType
    in_channels: int
    out_channels: int
    big: bool
    max_pool: bool
    bilinear: bool


@dataclass
class TrainConfig(AsDictDataclass, ModelParameters):  # pylint: disable=too-many-instance-attributes
    """Config for train segmentation model"""

    # dataset parameters
    image_dir: Union[str, Path]
    mask_dir: Union[str, Path]
    train_batch_size: int
    valid_batch_size: int
    valid_size: float
    # training parameters
    learning_rate: float
    seed: int
    num_epochs: int
    num_batch_steps: Optional[int]
    cpu: bool
    fp16: bool
    resume: Optional[str]
    # logging parameters
    logging: bool
    verbose: bool
    logdir: str
    name: str
    profile: bool


@dataclass
class EvaluationConfig(  # pylint: disable=too-many-instance-attributes
    AsDictDataclass, ModelParameters
):
    """Config for evaluation segmentation model"""

    # dataset parameters
    image_dir: Union[str, Path]
    mask_dir: Union[str, Path]
    eval_batch_size: int
    # eval parameters
    cpu: bool
    fp16: bool
    resume: Optional[str]
    # logging parameters
    logging: bool
    verbose: bool
    logdir: str
    name: str
    profile: bool


@dataclass
class InferenceConfig(ModelParameters):
    """Config for inference segmentation model"""

    pass  # pylint: disable=unnecessary-pass
