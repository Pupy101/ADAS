from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from .models import ModelSize, ModelType

CLASS_NAMES = ["main_road", "other_roads", "background"]


@dataclass
class AsDictDataclass:
    """Mixin class for factory method 'asdict'"""

    @staticmethod
    def _handle_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        return value

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Method for representation dataclass as dict"""
        exclude = exclude or set()
        return {k: self._handle_value(v) for k, v in asdict(self).items() if k not in exclude}


# model


@dataclass
class BaseModelParameters:
    """Model parameters for training segmentation model and pretraining encoder"""

    model: ModelType
    size: ModelSize
    in_channels: int
    max_pool: bool


@dataclass
class TEModelParameters(BaseModelParameters):
    """Model parameters for training encoder on classification task"""

    count_classes: int
    dropout: float


@dataclass
class TSModelParameters(BaseModelParameters):
    """Model parameters for training segmentation model"""

    out_channels: int
    bilinear: bool
    count_predict_masks: bool


# dataset & loader


@dataclass
class BaseDataParameters:
    """Data parameters for training segmentation model and pretraining encoder"""

    image_dir: Union[str, Path]
    valid_batch_size: int


@dataclass
class AddDataParameters:
    """Additional training parameters"""

    train_batch_size: int
    valid_size: float


@dataclass
class ESDataParameters(BaseDataParameters):
    """Data parameters for evaluation segmentation model"""

    mask_dir: Union[str, Path]


@dataclass
class TSDataParameters(ESDataParameters, AddDataParameters):
    """Data parameters for training segmentation model"""


# train


@dataclass
class BaseHyperparameters:
    """Hyperparameters for training segmentation model and pretraining encoder"""

    learning_rate: float
    seed: int
    num_epochs: int
    num_batch_steps: Optional[int]
    profile: bool


@dataclass
class TSHyperparameters(BaseHyperparameters):
    """Hyperparameters for training segmentation model"""

    predicts_coeffs: List[float]


# model weight


@dataclass
class BaseWeight:
    """Model weight for recovery model"""

    resume: Optional[str]


@dataclass
class TSWeight(BaseWeight):
    """Model weight for recovery segmentation model from it's weight or recovery only encoder"""

    resume_encoder: Optional[Union[str, Path]]

    def __post_init__(self):
        if self.resume and self.resume_encoder:
            raise ValueError("Please set only `resume` or `resume_encoder`")


# engine


@dataclass
class EngineParameters:
    """Engine parameters for training/evaluation"""

    cpu: bool
    fp16: bool


# logging


@dataclass
class LoggingParameters:
    """Logging parameters"""

    logdir: Union[str, Path]
    verbose: bool
    logging: bool
    name_run: str


# overall configs


@dataclass
class TrainEncoderConfig(  # pylint: disable=too-many-ancestors
    AsDictDataclass,
    TEModelParameters,
    BaseDataParameters,
    AddDataParameters,
    BaseHyperparameters,
    BaseWeight,
    EngineParameters,
    LoggingParameters,
):
    """
    Config of training model for classification images
    with encoder from segmentation model as backbone
    """


@dataclass
class EvaluationEncoderConfig(
    AsDictDataclass,
    TEModelParameters,
    BaseDataParameters,
    BaseWeight,
    EngineParameters,
    LoggingParameters,
):
    """
    Config of evaluation model for classification images
    with encoder from segmentation model as backbone
    """


@dataclass
class TrainSegmentationConfig(  # pylint: disable=too-many-ancestors
    AsDictDataclass,
    TSModelParameters,
    TSDataParameters,
    TSHyperparameters,
    TSWeight,
    EngineParameters,
    LoggingParameters,
):
    """Config of training segmentation model"""

    def __post_init__(self):
        super().__post_init__()
        assert self.count_predict_masks == len(
            self.predicts_coeffs
        ), "Count coefficients in `--predicts_coeffs` must be equal `--count_predict_masks`"


@dataclass
class EvaluationSegmentationConfig(  # pylint: disable=too-many-ancestors
    AsDictDataclass,
    TSModelParameters,
    ESDataParameters,
    BaseWeight,
    EngineParameters,
    LoggingParameters,
):
    """Config of evaluation segmentation model"""


Config = Union[
    EvaluationEncoderConfig,
    EvaluationSegmentationConfig,
    TrainEncoderConfig,
    TrainSegmentationConfig,
]
