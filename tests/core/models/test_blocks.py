from contextlib import nullcontext
from itertools import product
from typing import Optional

import pytest
import torch
from torch import Tensor

from adas.core.models.blocks import (
    RSUD1,
    RSUD5,
    DownsampleX2Block,
    DWConv2d,
    DWConv2dBNLReLU,
    DWConvT2d,
    DWConvT2dBNLReLU,
    UpsampleBlock,
    UpsampleX2Block,
)
from adas.core.models.types import DownsampleMode, UpsampleMode


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, dilation, size",
    product([3, 2], [3, 8], [2, 3], [1, 2], [0, 1], [1, 2], [112, 224]),
)
def test_dw_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
    size: int,
) -> None:
    model = DWConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == out_channels
    assert (
        output.size(2)
        == (input_.size(2) + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    )
    assert (
        output.size(3)
        == (input_.size(3) + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    )


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, size",
    product([3, 2], [3, 4], [3, 2], [1, 2], [0, 1], [0, 1], [1, 2], [112, 224]),
)
def test_dw_conv_t_2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int,
    dilation: int,
    size: int,
) -> None:
    model = DWConvT2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
    )

    context = (
        pytest.raises(RuntimeError)
        if output_padding >= stride and output_padding >= dilation
        else nullcontext()
    )
    input_ = torch.rand(2, in_channels, size, size)
    output = None
    with torch.no_grad(), context:  # type: ignore
        output: Tensor = model(input_)  # type: ignore
    if output is not None:
        assert input_.size(0) == output.size(0)
        assert output.size(1) == out_channels
        assert (
            output.size(2)
            == (input_.size(2) - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )
        assert (
            output.size(3)
            == (input_.size(3) - 1) * stride
            - 2 * padding
            + dilation * (kernel_size - 1)
            + output_padding
            + 1
        )


@pytest.mark.parametrize(
    "in_channels, out_channels, negative_slope, size",
    product([3, 2], [3, 4], [0.05, 0.1], [112, 224]),
)
def test_dw_conv2d_bn_lrelu(
    in_channels: int, out_channels: int, negative_slope: float, size: int
) -> None:
    model = DWConv2dBNLReLU(
        in_channels=in_channels,
        out_channels=out_channels,
        negative_slope=negative_slope,
    )
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == out_channels
    assert input_.size(2) == output.size(2)
    assert input_.size(3) == output.size(3)


@pytest.mark.parametrize("in_channels, out_channels, size", product([3, 2], [3, 4], [112, 224]))
def test_dw_conv_t_2d_bn_lrelu(in_channels: int, out_channels: int, size: int) -> None:
    model = DWConvT2dBNLReLU(in_channels=in_channels, out_channels=out_channels)
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == out_channels
    assert input_.size(2) * 2 == output.size(2)
    assert input_.size(3) * 2 == output.size(3)


@pytest.mark.parametrize("mode, in_channels, size", product(UpsampleMode, [3, 4], [120, 224]))
def test_upsamplex2(mode: UpsampleMode, in_channels: int, size: int):
    model = UpsampleX2Block(mode=mode.value, in_channels=in_channels)
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == in_channels
    assert input_.size(2) * 2 == output.size(2)
    assert input_.size(3) * 2 == output.size(3)


@pytest.mark.parametrize("scale_factor, in_channels, size", product([2, 4, 6], [3, 4], [55, 224]))
def test_upsample(scale_factor: int, in_channels: int, size: int):
    model = UpsampleBlock(scale_factor=scale_factor)
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == in_channels
    assert input_.size(2) * scale_factor == output.size(2)
    assert input_.size(3) * scale_factor == output.size(3)


@pytest.mark.parametrize("mode, in_channels, size", product(DownsampleMode, [3, 4], [224, 330]))
def test_downsample(mode: DownsampleMode, in_channels: int, size: int):
    model = DownsampleX2Block(mode=mode.value, in_channels=in_channels)
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == in_channels
    assert input_.size(2) // 2 == output.size(2)
    assert input_.size(3) // 2 == output.size(3)


@pytest.mark.parametrize(
    "in_channels, out_channels, mid_channels, depth, size",
    product([3, 4], [4, 8], [2, 16, None], [4, 5], [224, 512]),
)
def test_rsud1(
    in_channels: int,
    out_channels: int,
    mid_channels: Optional[int],
    depth: int,
    size: int,
):
    model = RSUD1(
        in_channels=in_channels,
        out_channels=out_channels,
        mid_channels=mid_channels,
        depth=depth,
    )
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == out_channels
    assert input_.size(2) == output.size(2)
    assert input_.size(3) == output.size(3)


@pytest.mark.parametrize(
    "in_channels, out_channels, mid_channels, size",
    product([3, 4], [4, 8], [2, 16, None], [112, 224]),
)
def test_rsud5(in_channels: int, out_channels: int, mid_channels: Optional[int], size: int):
    model = RSUD5(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
    input_ = torch.rand(2, in_channels, size, size)
    with torch.no_grad():
        output: Tensor = model(input_)
    assert input_.size(0) == output.size(0)
    assert output.size(1) == out_channels
    assert input_.size(2) == output.size(2)
    assert input_.size(3) == output.size(3)
