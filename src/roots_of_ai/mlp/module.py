"""Base module for multilayer perceptron components."""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from .parameter import Parameter

__all__ = ["Module"]


class Module:
    """Base class for composable neural-network modules."""

    def parameters(self) -> Iterable[Parameter]:
        """Returns trainable parameters."""
        return ()

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the module output for the given inputs."""
        raise NotImplementedError(f"Module {type(self).__name__} must implement the 'forward' method")

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Propagate an output gradient to the inputs of the module."""
        raise NotImplementedError(f"Module {type(self).__name__} must implement the 'backward' method")

    def __call__(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Call the module with the given inputs."""
        return self.forward(inputs)
