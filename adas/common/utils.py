from typing import Tuple, TypeVar

from torch import Generator
from torch.utils.data import Dataset, Subset, random_split

CustomDataset = TypeVar("CustomDataset", bound=Dataset)


def train_test_split(
    dataset: CustomDataset, test_size: float, seed: int = 1234
) -> Tuple[Subset, Subset]:
    assert 0 < test_size < 1, "Test size must be in interval (0, 1)"
    dataset_length = len(dataset)  # type: ignore
    test_size = round(dataset_length * test_size)
    train_size = dataset_length - test_size
    train, test = random_split(
        dataset, lengths=[train_size, test_size], generator=Generator().manual_seed(seed)
    )
    return train, test
