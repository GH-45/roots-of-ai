from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .parameter import Parameter

__all__ = ["Module"]


class Module:
    def parameters(self) -> Iterable[Parameter]:
        return ()

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(f"Module {type(self).__name__} must implement the 'forward' method")

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError(f"Module {type(self).__name__} must implement the 'backward' method")

    def __call__(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(inputs)
