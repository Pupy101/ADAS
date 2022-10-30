import random
from typing import List, Tuple, TypeVar

T = TypeVar("T")


def train_test_split(data: List[T], test_size: float, seed: int = 1234) -> Tuple[List[T], List[T]]:
    """Split sequence of items on train/val subsequences"""
    random.seed(seed)
    train_data: List[T] = []
    test_data: List[T] = []
    for item in data:
        if random.random() < test_size:
            test_data.append(item)
        else:
            train_data.append(item)
    return train_data, test_data
