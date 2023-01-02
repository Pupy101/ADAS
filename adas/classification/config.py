from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, Union

from adas.core.config import AsDictDataclassMixin, EngineParams, HyperParams, WeightLoggingParams
from adas.core.models.types import DownsampleMode

from .models.types import ModelSize, ModelType


@dataclass
class ModelParams:
    """Model parameters for training classification model"""

    model: ModelType
    in_channels: int
    size: ModelSize
    downsample: DownsampleMode
    classes: int


@dataclass
class EvalDataParams:
    """Data parameters for evaluation classification model"""

    data_dir: Union[str, Path]
    valid_batch_size: int


@dataclass
class TrainDataParams(EvalDataParams):
    """Data parameters for training segmentation model"""

    train_batch_size: int
    valid_size: float


@dataclass
class TrainCfg(  # pylint: disable=too-many-ancestors
    AsDictDataclassMixin,
    ModelParams,
    TrainDataParams,
    HyperParams,
    WeightLoggingParams,
    EngineParams,
):
    """Config of training segmentation model"""


@dataclass
class EvalCfg(
    AsDictDataclassMixin,
    ModelParams,
    EvalDataParams,
    WeightLoggingParams,
    EngineParams,
):
    """Config of evaluation segmentation model"""


Config = TypeVar("Config", TrainCfg, EvalCfg)
