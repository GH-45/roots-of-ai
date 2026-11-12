"""Basic structural block composed of a group of neurons."""

import numpy as np
from numpy.typing import NDArray

from .module import Module
from .parameter import Parameter

__all__ = ["DenseLayer"]


class DenseLayer(Module):
    """Fully connected layer.

    Args:
        in_features: Number of input features of the layer.
        out_features: Number of output features of the layer.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self._in_features = in_features
        self._out_features = out_features

        self._inputs: NDArray[np.float64] | None = None

        # Trainable parameters
        self._weights = Parameter(np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features))
        self._bias = Parameter(np.zeros((1, out_features)))

    def parameters(self) -> list[Parameter]:
        """Return the trainable weights and bias."""
        return [self._weights, self._bias]

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes the weighted sum of the input features plus the bias.

        Args:
            inputs: Input feature vector.

        Returns:
            Result of the weighted sum.
        """
        # Save inputs for gradient computation in the backward pass
        self._inputs = inputs

        return self._inputs @ self._weights.data + self._bias.data

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Accumulate parameter gradients and return the input gradient.

        Args:
            out_gradient: Gradient of the loss with respect to the output of the layers.

        Returns:
            Gradient of the loss with respect to the layer inputs.
        """
        if self._inputs is None:
            raise RuntimeError("Cannot perform backward pass before forward pass. Call forward() first")

        self._weights.grad = self._inputs.T @ out_gradient
        self._bias.grad = np.sum(out_gradient, axis=0, keepdims=True)

        return out_gradient @ self._weights.data.T

    def __repr__(self) -> str:
        """Return a string representation of the layer."""
        return f"DenseLayer(in_features={self._in_features}, out_features={self._out_features})"
