from dataclasses import dataclass, fields
from typing import Optional, Tuple, Type

from torch.nn import Module


@dataclass
class TorchModule:
    module: Type[Module]

    def create_module(self) -> Module:
        module = self.module
        kwargs = {field.name: getattr(self, field.name) for field in fields(self) if field.name != "module"}
        return module(**kwargs)


@dataclass
class DWConv2dConfig(TorchModule):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1


@dataclass
class DWConv2dSConfig(DWConv2dConfig):
    pass


@dataclass
class DWConv2dBNLReLUConfig(TorchModule):
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    padding: int = 1
    dilation: int = 1
    negative_slope: float = 0.05


@dataclass
class DWConvT2dConfig(TorchModule):
    in_channels: int
    out_channels: int
    kernel_size: int = 2
    stride: int = 2
    padding: int = 0
    output_padding: int = 0
    dilation: int = 1


@dataclass
class DWConvT2dBNLReLUConfig(DWConvT2dConfig):
    negative_slope: float = 0.05


@dataclass
class UpDWConv2dSConfig(TorchModule):
    scale: int
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    dilation: int = 1


@dataclass
class BlockWithMiddleChannels(TorchModule):
    in_channels: int
    out_channels: int
    mid_channels: Optional[int] = None


@dataclass
class UpBlockConfig(BlockWithMiddleChannels):
    bilinear: bool = True


@dataclass
class DownBlockConfig(BlockWithMiddleChannels):
    max_pool: bool = True


@dataclass
class RSUD1Config(BlockWithMiddleChannels):
    kernel_size: int = 3
    padding: int = 1
    dilation: int = 2
    pad_dilation: int = 2
    depth: int = 5
    max_pool: bool = True
    bilinear: bool = True


@dataclass
class RSUD5Config(BlockWithMiddleChannels):
    kernel_size: int = 3
    padding: int = 1
    dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2)
    pad_dilations: Tuple[int, int, int, int, int] = (2, 4, 8, 4, 2)
