from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Union

from catalyst.callbacks import Callback
from catalyst.core.logger import ILogger
from torch import nn, optim
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
class TrainDDPConfig(AsDictDataclass):
    datasets: Mapping[str, DatasetArgs]
    train_batch_size: int
    valid_batch_size: int

    def __post_init__(self) -> None:
        assert "train" in self.datasets
        assert "valid" in self.datasets


@dataclass
class InferenceConfig(AsDictDataclass):
    pass
