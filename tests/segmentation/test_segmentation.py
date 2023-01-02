from itertools import product
from typing import Union

import pytest
import torch

from adas.core.models.types import DownsampleMode, UpsampleMode
from adas.segmentation.models.types import ModelSize
from adas.segmentation.models.u2net import U2net
from adas.segmentation.models.unet import Unet


@pytest.mark.parametrize(
    "model_type, in_channels, out_channels, size, downsample_mode, upsample_mode, count_features",
    product([Unet, U2net], [3], [2, 3], ModelSize, DownsampleMode, UpsampleMode, [1, 2, 4]),
)
def test_unet(  # pylint: disable=too-many-arguments
    model_type: Union[Unet, U2net],
    in_channels: int,
    out_channels: int,
    size: ModelSize,
    downsample_mode: DownsampleMode,
    upsample_mode: UpsampleMode,
    count_features: int,
):
    """Testing of Unet model forward pass"""
    model = model_type(
        in_channels=in_channels,
        out_channels=out_channels,
        size=size.value,
        downsample_mode=downsample_mode.value,
        upsample_mode=upsample_mode.value,
        count_features=count_features,
    )
    input_batch = torch.rand(2, in_channels, 224, 224)
    with torch.no_grad():
        outputs_batch = model(input_batch)
    assert len(outputs_batch) == count_features
    for output in outputs_batch:
        assert tuple(output.shape) == (2, out_channels, 224, 224)
