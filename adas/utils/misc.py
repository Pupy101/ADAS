import random
from enum import Enum
from typing import Any, List, Tuple, Type, TypeVar

T = TypeVar("T")

EnumType = TypeVar("EnumType", bound=Enum)  # pylint: disable=invalid-name


def train_test_split(data: List[T], test_size: float, seed: int = 1234) -> Tuple[List[T], List[T]]:
    """Split sequence of items on train/val subsequences"""
    random.seed(seed)
    assert 0 < test_size < 1, "test_size must in interval (0, 1)"
    train_data: List[T] = []
    test_data: List[T] = []
    for item in data:
        if random.random() < test_size:
            test_data.append(item)
        else:
            train_data.append(item)
    return train_data, test_data


def find_enum(value: Any, enum_type: Type[EnumType]) -> EnumType:
    """Find enum by it value or raise ValueError"""
    for elem in enum_type:
        if value == elem.value:
            return elem
    raise ValueError(f"Strange value {value} for enum {enum_type}")
