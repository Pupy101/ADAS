import torch

from typing import Union, Tuple

from torch import Tensor
from torch.nn import functional as F
from numpy import ndarray


def upsample(image: Tensor, shape: Union[Tuple, Tensor]) -> Tensor:
    """
    Function upsampling input image to needed shape:
    ------------------------------------------------
    Example:
        upsample(torch.rand(1, 3, 200, 200), [400, 400]) -> Tensor with shape [1, 3, 400, 400]
    ------------------------------------------------
    """
    assert len(shape) == 2, 'Parameter shape must be sequence with legth equal 2'
    return F.interpolate(
        image,
        size=shape,
        mode='bilinear',
        align_corners=False
    )


def upsampleX2(image: Tensor):
    """
    Function upsampling input image twice time:
    ------------------------------------------------
    Example:
        upsampleX2(torch.rand(1, 3, 100, 100)) -> Tensor with shape [1, 3, 200, 200]
    ------------------------------------------------
    """
    shape = image.shape[-2:]
    return upsample(image, [shape[0]*2, shape[1]*2])


def upsample_like(image: Tensor, example: Tensor):
    """
    Function changing input image shape to example shape
    ------------------------------------------------
    Example:
        upsample_like(torch.rand(1, 3, 50, 50), torch.rand(1, 3, 200, 200)) -> Tensor with shape [1, 3, 200, 200]
    """
    shape = example.shape[-2:]
    return upsample(image, shape)



def transform_tensor_to_numpy(batch: Tensor) -> ndarray:
    """
    Function changing input tensor batch in numpy array with round to 0 or 1
    ----------------------------------------------
    Example:
        transform_tensor_to_numpy(torch.ones(1, 3))
    """
    assert len(batch.shape) == 4, 'Input batch must have 4 dimensions [B, C, H, W]'
    return torch.round(batch.permute(0, 2, 3, 1)).detach().cpu().numpy()
