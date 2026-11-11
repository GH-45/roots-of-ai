from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

__all__ = ["Parameter"]


@dataclass
class Parameter:
    data: NDArray[np.float64]
    grad: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        self.grad = np.zeros_like(self.data)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)
