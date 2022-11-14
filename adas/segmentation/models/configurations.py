from dataclasses import dataclass, fields
from enum import Enum
from typing import List, Optional, Tuple, Type

from torch.nn import Module, Sequential

from .blocks import *  # pylint: disable=unused-wildcard-import,wildcard-import


class ModelSize(Enum):
    """Model sizes"""

    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    BIG = "BIG"


@dataclass
class TorchModuleConfig:
    """Mixin class factory method for initialize module"""

    def create(self) -> Module:
        """Method for initialize module"""
        assert hasattr(self, "module")
        self.module: Type[Module]  # pylint: disable=attribute-defined-outside-init
        module = self.module
        kwargs = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "module"
        }
        return module(**kwargs)


@dataclass
class DWConv2dConfig(TorchModuleConfig):
    """Config of DWConv2d module"""

    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    module: Type[Module] = DWConv2d


@dataclass
class DWConv2dSigmoidConfig(DWConv2dConfig):
    """Config of DWConv2dSigmoid module"""

    module: Type[Module] = DWConv2dSigmoid


@dataclass
class DWConv2dBNLReLUConfig(DWConv2dConfig):
    """Config of DWConv2dBNLReLU module"""

    negative_slope: float = 0.1
    module: Type[Module] = DWConv2dBNLReLU


@dataclass
class DWConvT2dConfig(TorchModuleConfig):  # pylint: disable=too-many-instance-attributes
    """Config of DWConvT2d module"""

    in_channels: int
    out_channels: int
    kernel_size: int = 2
    stride: int = 2
    padding: int = 0
    output_padding: int = 0
    dilation: int = 1
    module: Type[Module] = DWConvT2d


@dataclass
class DWConvT2dBNLReLUConfig(DWConvT2dConfig):
    """Config of DWConvT2dBNLReLU module"""

    negative_slope: float = 0.05
    module: Type[Module] = DWConvT2dBNLReLU


@dataclass
class DWConv2dSigmoidUpConfig(TorchModuleConfig):  # pylint: disable=too-many-instance-attributes
    """Config of DWConv2dSigmoidUp module"""

    scale: int
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    module: Type[Module] = DWConv2dSigmoidUp


@dataclass
class UpBlockConfig(TorchModuleConfig):
    """Config of UpBlock module"""

    in_channels: int
    bilinear: bool = True
    module: Type[Module] = UpBlock


@dataclass
class DownBlockConfig(TorchModuleConfig):
    """Config of DownBlock module"""

    in_channels: int
    max_pool: bool = True
    module: Type[Module] = DownBlock


@dataclass
class RSUD1Config(TorchModuleConfig):  # pylint: disable=too-many-instance-attributes
    """Config of RSUD1 module"""

    in_channels: int
    out_channels: int
    mid_channels: Optional[int] = None
    kernel_size: int = 3
    padding: int = 1
    dilation: int = 2
    pad_dilation: int = 2
    depth: int = 5
    max_pool: bool = True
    bilinear: bool = True
    module: Type[Module] = RSUD1


@dataclass
class RSUD5Config(TorchModuleConfig):  # pylint: disable=too-many-instance-attributes
    """Config of RSUD5 module"""

    in_channels: int
    out_channels: int
    mid_channels: Optional[int] = None
    kernel_size: int = 3
    padding: int = 1
    dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2)
    pad_dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2)
    module: Type[Module] = RSUD5


@dataclass
class ManyConfigs(TorchModuleConfig):
    """Many TorchModuleConfig-s sequence"""

    configs: List[DWConv2dBNLReLUConfig]

    def create(self) -> Module:
        """Initialize sequence of modules as torch.nn.Sequential"""
        return Sequential(*[c.create() for c in self.configs])
