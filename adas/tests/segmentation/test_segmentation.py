from itertools import product
from typing import Union

import pytest
import torch

from adas.segmentation.models import ModelSize, U2net, Unet


@pytest.mark.parametrize(
    "model_type, in_channels, out_channels, size, max_pool, bilinear, count_predict_masks",
    product([Unet, U2net], [3], [2, 3], ModelSize, [True, False], [True, False], [1, 2, 4]),
)
def test_unet(  # pylint: disable=too-many-arguments
    model_type: Union[Unet, U2net],
    in_channels: int,
    out_channels: int,
    size: ModelSize,
    max_pool: bool,
    bilinear: bool,
    count_predict_masks: int,
):
    """Testing of Unet model forward pass"""
    model = model_type(
        in_channels=in_channels,
        out_channels=out_channels,
        size=size,
        max_pool=max_pool,
        bilinear=bilinear,
        count_predict_masks=count_predict_masks,
    )
    input_batch = torch.rand(2, in_channels, 224, 224)
    with torch.no_grad():
        outputs_batch = model(input_batch)
    assert len(outputs_batch) == count_predict_masks
    for output in outputs_batch:
        assert tuple(output.shape) == (2, out_channels, 224, 224)
