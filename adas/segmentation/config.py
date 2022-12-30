from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TypeVar, Union

from adas.core.models.types import DownsampleMode, UpsampleMode

from .models.types import ModelSize, ModelType

CLASS_NAMES = ["main_road", "other_roads", "background"]


@dataclass
class AsDictDataclassMixin:
    """Mixin class for factory method 'asdict'"""

    @staticmethod
    def handle(obj: Any) -> Any:
        """Handle for get Enum value"""
        if isinstance(obj, Enum):
            return obj.value
        return obj

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Method for representation dataclass as dict"""
        exclude = exclude or set()
        return {k: self.handle(v) for k, v in asdict(self).items() if k not in exclude}


@dataclass
class ModelParams:
    """Model parameters for training segmentation model"""

    model: ModelType
    in_channels: int
    out_channels: int
    size: ModelSize
    downsample: DownsampleMode
    upsample: UpsampleMode
    count_features: int


@dataclass
class EvalDataParams:
    """Data parameters for evaluation segmentation model"""

    image_dir: Union[str, Path]
    mask_dir: Union[str, Path]
    valid_batch_size: int


@dataclass
class TrainDataParams(EvalDataParams):
    """Data parameters for training segmentation model"""

    train_batch_size: int
    valid_size: float


@dataclass
class EvalHyperParams:
    """Hyperparameters for evaluation segmentation model"""

    predicts_coeffs: List[float]


@dataclass
class TrainHyperParams(EvalHyperParams):
    """Hyperparameters for training segmentation model"""

    learning_rate: float
    seed: int
    num_epochs: int
    num_batch_steps: Optional[int]
    profile: bool


@dataclass
class EvalWeightParams:
    """Model weight for recovery model"""

    resume: Optional[str]


@dataclass
class TrainWeightParams(EvalWeightParams):
    """Model weight for recovery model"""

    resume_encoder: Optional[Union[str, Path]]

    def __post_init__(self):
        if self.resume and self.resume_encoder:
            raise ValueError("Please set only `resume` or `resume_encoder`")


@dataclass
class EngineParameters:
    """Engine parameters for training/evaluation"""

    cpu: bool
    fp16: bool


@dataclass
class LoggingParameters:
    """Logging parameters"""

    logdir: Union[str, Path]
    verbose: bool
    wandb: bool
    name_run: str


# overall configs


@dataclass
class TrainCfg(  # pylint: disable=too-many-ancestors
    AsDictDataclassMixin,
    ModelParams,
    TrainDataParams,
    TrainHyperParams,
    TrainWeightParams,
    EngineParameters,
    LoggingParameters,
):
    """Config of training segmentation model"""


@dataclass
class EvalCfg(
    AsDictDataclassMixin,
    ModelParams,
    EvalDataParams,
    EvalHyperParams,
    EvalWeightParams,
    EngineParameters,
    LoggingParameters,
):
    """Config of evaluation segmentation model"""


Config = TypeVar("Config", TrainCfg, EvalCfg)
