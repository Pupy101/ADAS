from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Set, Union

from torch.utils.data import DataLoader


@dataclass
class AsDictDataclass:
    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Mapping[str, Any]:
        dictionary = asdict(self)
        if exclude is not None:
            for key in exclude:
                dictionary.pop(key)
        return dictionary


@dataclass
class DatasetArgs(AsDictDataclass):
    image_dir: Union[str, Path]
    mask_dir: Union[str, Path]
    transforms: Callable
    image_extensions: Optional[Set[str]] = None


@dataclass
class DDPConfig(AsDictDataclass):
    datasets: Mapping[str, DatasetArgs]
    train_batch_size: int
    valid_batch_size: int

    def __post_init__(self) -> None:
        assert "train" in self.datasets
        assert "valid" in self.datasets


class ModelType(Enum):
    UNET = "UNET"
    U2NET = "U2NET"


@dataclass
class TrainConfig(AsDictDataclass):
    model: ModelType
    in_channels: int
    out_channels: int
    big: bool
    max_pool: bool
    bilinear: bool
    seed: int
    num_epochs: int
    logdir: str
    resume: Optional[str] = None
    valid_loader: str = "valid"
    valid_metric: str = "loss"
    verbose: bool = True
    cpu: bool = True
    fp16: bool = True
    ddp: bool = False
    minimize_valid_metric: bool = True
    loaders: Optional[Mapping[str, DataLoader]] = None


@dataclass
class InferenceConfig(AsDictDataclass):
    pass
