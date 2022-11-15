from itertools import product

import pytest
import torch

from adas.segmentation.models import ModelSize, Unet


@pytest.mark.parametrize(
    "in_channels, out_channels, size, max_pool, bilinear",
    product([3], [2, 3], ModelSize, [True, False], [True, False]),
)
def test_unet(in_channels: int, out_channels: int, size: ModelSize, max_pool: bool, bilinear: bool):
    model = Unet(
        in_channels=in_channels,
        out_channels=out_channels,
        size=size,
        max_pool=max_pool,
        bilinear=bilinear,
    )
    input_batch = torch.rand(16, in_channels, 224, 224)
    with torch.no_grad():
        outputs_batch = model(input_batch)
    for output in outputs_batch:
        assert tuple(output.shape) == (16, out_channels, 224, 224)
