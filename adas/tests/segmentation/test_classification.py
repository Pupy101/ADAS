from itertools import product
from typing import Union

import pytest
import torch

from adas.segmentation.models import Classificator, ModelSize
from adas.segmentation.models.u2net import U2netEncoder
from adas.segmentation.models.unet import UnetEncoder
from adas.segmentation.utils.configs import ModelType


@pytest.mark.parametrize(
    "feature_extractor_type, in_channels, size, max_pool, count_classes, dropout",
    product(ModelType, [3], ModelSize, [True, False], [3, 100], [0.1, 0.4]),
)
def test_classificator(
    feature_extractor_type: ModelType,
    in_channels: int,
    size: ModelSize,
    max_pool: bool,
    count_classes: int,
    dropout: float,
):
    feature_extractor: Union[UnetEncoder, U2netEncoder]
    if feature_extractor_type is ModelType.UNET:
        feature_extractor = UnetEncoder(in_channels=in_channels, size=size, max_pool=max_pool)
    elif feature_extractor_type is ModelType.U2NET:
        feature_extractor = U2netEncoder(in_channels=in_channels, size=size, max_pool=max_pool)
    else:
        raise ValueError(f"Strange device type: {feature_extractor_type}")
    model = Classificator(
        feature_extractor=feature_extractor, count_classes=count_classes, dropout=dropout
    )
    input_batch = torch.rand(4, in_channels, 224, 224)
    with torch.no_grad():
        outputs_batch = model(input_batch)
    assert tuple(outputs_batch.shape) == (4, count_classes)
