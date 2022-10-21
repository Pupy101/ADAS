from typing import Tuple, Union

from torch import Tensor
from torch.nn import Module


class AggregatorManyOutputsLoss(Module):
    def __init__(self, losses: Union[Module, Tuple[Module, ...]], coefficients: Tuple[Union[float, int], ...]) -> None:
        super().__init__()
        self.losses: Tuple[Module, ...]
        if isinstance(losses, Module):
            self.losses = (losses,)
        else:
            self.losses = losses
        self.coeffs = coefficients

    def forward(self, outputs: Tuple[Tensor, ...], targets: Tensor) -> Tensor:
        assert len(outputs) == len(self.coeffs)
        return sum(loss(o, targets) * c for loss in self.losses for o, c in zip(outputs, self.coeffs))
