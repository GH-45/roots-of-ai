import numpy as np
from numpy.typing import NDArray

from .module import Module
from .parameter import Parameter

__all__ = ["DenseLayer"]


class DenseLayer(Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        self._in_features = in_features
        self._out_features = out_features

        self._inputs: NDArray[np.float64] | None = None
        self._weight = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self._bias = Parameter(np.zeros((1, out_features)))

    def parameters(self) -> list[Parameter]:
        return [self._weight, self._bias]

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        self._inputs = inputs

        return self._inputs @ self._weight.data + self._bias.data

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._inputs is None:
            raise RuntimeError("Cannot perform backward pass before forward pass. Call forward() first")

        self._weight.grad = self._inputs.T @ out_gradient
        self._bias.grad = np.sum(out_gradient, axis=0, keepdims=True)

        return out_gradient @ self._weight.data.T
