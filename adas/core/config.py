from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union


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
class HyperParams:
    """Hyperparameters for training model"""

    learning_rate: float
    seed: int
    num_epochs: int
    num_batch_steps: Optional[int]
    profile: bool


@dataclass
class WeightParams:
    """Weight parameters for recovery model"""

    resume: Optional[str]


@dataclass
class LoggingParams:
    """Logging parameters"""

    logdir: Union[str, Path]
    verbose: bool
    wandb: bool
    name_run: str


@dataclass
class WeightLoggingParams(WeightParams, LoggingParams):
    """Weight and logging parameters"""


@dataclass
class EngineParams:
    """Engine parameters for training/evaluation"""

    cpu: bool
    fp16: bool
