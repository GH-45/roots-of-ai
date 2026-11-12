"""Trainable parameter representation used by multilayer perceptron modules."""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["Parameter"]


@dataclass
class Parameter:
    """Trainable array and its accumulated gradient.

    Args:
        data: Parameter values.
        grad: Accumulated gradient for the parameter.
    """

    data: NDArray[np.float64]
    grad: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:  # noqa: D105
        # Initialize the gradient with the same shape as the data
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        """Reset the accumulated gradient."""
        self.grad.fill(0.0)
