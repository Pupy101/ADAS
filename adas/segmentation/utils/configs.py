from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Set, Union

from torch.utils.data import DataLoader

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


@dataclass
class DatasetArgs(AsDictDataclass):
    """Params for create torch Dataset (need for DDP)"""

    image_dir: Union[str, Path]
    mask_dir: Union[str, Path]
    transforms: Callable
    image_extensions: Optional[Set[str]] = None


@dataclass
class DDPConfig(AsDictDataclass):
    """DDP special config for create dataloader in each process"""

    dataset: DatasetArgs
    train_batch_size: int
    valid_batch_size: int
    seed: int
    valid_size: float


class ModelType(Enum):
    """Acceptable models types"""

    UNET = "UNET"
    U2NET = "U2NET"


@dataclass
class TrainConfig(AsDictDataclass):  # pylint: disable=too-many-instance-attributes
    """Config for train segmentation model"""

    model: ModelType
    in_channels: int
    out_channels: int
    big: bool
    max_pool: bool
    bilinear: bool
    learning_rate: float
    seed: int
    train_batch_size: int
    valid_batch_size: int
    num_epochs: int
    logging: bool
    verbose: bool
    logdir: str
    valid_size: float
    cpu: bool
    fp16: bool
    ddp: bool
    resume: Optional[str]
    valid_loader: str = "valid"
    valid_metric: str = "loss"
    minimize_valid_metric: bool = True
    count_batches: Optional[int] = None  # for debugging
    loaders: Optional[Mapping[str, DataLoader]] = None  # for ddp


@dataclass
class InferenceConfig(AsDictDataclass):
    """Config for inference segmentation model"""

    pass  # pylint: disable=unnecessary-pass


__all__ = ["CLASS_NAMES", "DatasetArgs", "DDPConfig", "ModelType", "TrainConfig", "InferenceConfig"]
