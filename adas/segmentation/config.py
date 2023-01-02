from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TypeVar, Union

from adas.core.config import AsDictDataclassMixin, EngineParams, HyperParams, WeightLoggingParams
from adas.core.models.types import DownsampleMode, UpsampleMode

from .models.types import ModelSize, ModelType

CLASS_NAMES = ["main_road", "other_roads", "background"]


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
class TrainWeightLoggingParams(WeightLoggingParams):

    resume_encoder: Optional[Union[str, Path]]

    def __post_init__(self):
        if self.resume and self.resume_encoder:
            raise ValueError("Please set only `resume` or `resume_encoder`")


@dataclass
class TrainCfg(  # pylint: disable=too-many-ancestors
    AsDictDataclassMixin,
    ModelParams,
    TrainDataParams,
    HyperParams,
    EvalHyperParams,
    TrainWeightLoggingParams,
    EngineParams,
):
    """Config of training segmentation model"""


@dataclass
class EvalCfg(
    AsDictDataclassMixin,
    ModelParams,
    EvalDataParams,
    EvalHyperParams,
    WeightLoggingParams,
    EngineParams,
):
    """Config of evaluation segmentation model"""


Config = TypeVar("Config", TrainCfg, EvalCfg)
