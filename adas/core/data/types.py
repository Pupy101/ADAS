from enum import Enum


class DatasetType(Enum):
    """Type of dataset"""

    TRAIN = "TRAIN"
    VALID = "VALID"
    TEST = "TEST"
