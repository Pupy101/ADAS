from itertools import product
from typing import Type, Union

import pytest
import torch

from adas.classification.models.segmentation import U2netEncoderClassifier, UnetEncoderClassifier
from adas.classification.models.types import DownsampleMode, ModelSize


@pytest.mark.parametrize(
    "model_type, in_channels, size, downsample_mode, count_classes",
    product(
        [UnetEncoderClassifier, U2netEncoderClassifier],
        [3, 4],
        [_.value for _ in ModelSize],
        [_.value for _ in DownsampleMode],
        [100, 1000],
    ),
)
def test_unet_encoder_classifier(
    model_type: Type[Union[U2netEncoderClassifier, UnetEncoderClassifier]],
    in_channels: int,
    size: str,
    downsample_mode: str,
    count_classes: int,
):
    model = model_type(
        in_channels=in_channels,
        size=size,
        downsample_mode=downsample_mode,
        count_classes=count_classes,
    )
    input_ = torch.rand(2, in_channels, 224, 224)
    with torch.no_grad():
        output = model(input_)
    assert len(output.shape) == 2
    assert output.size(1) == count_classes
