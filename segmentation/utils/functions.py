from torch.nn import functional as F


def UpsampleX2(image):
    """
    Unit with structure: Upsample X2
    """
    shape = image.shape[2:]
    return F.interpolate(image, size=[shape[0]*2, shape[1]*2], mode='bilinear', align_corners=False)
