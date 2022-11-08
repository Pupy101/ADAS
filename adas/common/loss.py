from typing import Optional, Tuple, Union

from torch import Tensor
from torch.nn import Module


class AggregatorManyOutputsLoss(Module):
    """Aggregator for many losses and many outputs from module"""

    def __init__(
        self,
        losses: Union[Module, Tuple[Module, ...]],
        coefficients: Tuple[Union[float, int], ...],
        losses_coefficients: Optional[Tuple[Union[float, int], ...]] = None,
    ) -> None:
        super().__init__()
        self.losses: Tuple[Module, ...]
        if isinstance(losses, Module):
            self.losses = (losses,)
        else:
            self.losses = losses
        self.coeffs = coefficients
        if losses_coefficients is None:
            losses_coefficients = tuple(1 for _ in range(len(self.losses)))
        self.losses_coefficients: Tuple[Union[float, int], ...] = losses_coefficients
        assert len(self.losses_coefficients) == len(
            self.losses
        ), "Set right coefficients losses for AggregatorManyOutputsLoss"

    def forward(self, outputs: Tuple[Tensor, ...], targets: Tensor) -> Tensor:
        """Forward step of module"""
        assert len(outputs) == len(
            self.coeffs
        ), "Set right coefficients outputs for AggregatorManyOutputsLoss"
        return sum(
            loss(o, targets) * c * l_c
            for loss, l_c in zip(self.losses, self.losses_coefficients)
            for o, c in zip(outputs, self.coeffs)
        )
