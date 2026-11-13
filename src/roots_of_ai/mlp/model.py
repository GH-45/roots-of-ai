"""MLP (Multilayer Perceptron) model implementation.

This implementation consists of a sequence of layers that can be trained end-to-end.
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from .module import Module
from .parameter import Parameter

__all__ = ["Sequential"]


def _noop_callback(_epoch: int, _model: Any) -> bool:
    """A no-operation callback function that does nothing."""
    return False


class Loss(Protocol):
    """Interface for loss functions used to train a model."""

    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> np.float64:
        """Compute the loss value within the prediction and true value."""
        ...

    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the gradient of the loss with respect to the predictions."""
        ...


class Optimizer(Protocol):
    """Interface for optimizers that update model parameters."""

    def step(self) -> None:
        """Update parameters using their accumulated gradients."""
        ...

    def zero_grad(self) -> None:
        """Reset the gradients after an update."""
        ...


class Sequential(Module):
    """A model that chains modules and trains them end-to-end.

    Args:
        layers: Modules to apply in order.

    Compile the model with a loss and optimizer before calling :meth:`fit`.
    """

    def __init__(self, layers: Sequence[Module]) -> None:
        self._layers = list(layers)

        self._loss: Loss | None = None
        self._optimizer: Optimizer | None = None

    def parameters(self) -> list[Parameter]:
        """Returns trainable parameters."""
        return [parameter for layer in self._layers for parameter in layer.parameters()]

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Propagate inputs through the layers in order.

        Args:
            inputs: Input feature vector.

        Returns:
            Output feature vector.
        """
        output = inputs

        for layer in self._layers:
            output = layer.forward(output)

        return output

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        """Propagate an output gradient through the layers in reverse order.

        Args:
            out_gradient: Gradient of the loss with respect to the output of the layers.

        Returns:
            Gradient of the loss with respect to the layer inputs.
        """
        gradient = out_gradient

        for layer in reversed(self._layers):
            gradient = layer.backward(gradient)

        return gradient

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Perform a prediction for a given input vector.

        Args:
            inputs: Input feature vector.

        Returns:
            Model prediction.
        """
        return self.forward(inputs)

    def compile(self, *, optimizer: Optimizer, loss: Loss) -> None:
        """Configure the optimizer and loss used by :meth:`fit`.

        Args:
            optimizer: Optimizer to use for training.
            loss: Loss function to use for training.
        """
        self._optimizer = optimizer
        self._loss = loss

    def fit(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        *,
        epochs: int = 100,
        callback: Callable[[int, Self], bool] = _noop_callback,
    ) -> Self:
        """Fit the model using Batch Gradient Descent and Backpropagation.

        Args:
            x_train: Training samples.
            y_train: Target values.
            epochs: Maximum number of training iterations.
            callback: Function called at the end of each epoch.
                      Returns ``True`` to stop training early.

        Returns:
            Trained model.
        """
        if self._optimizer is None or self._loss is None:
            raise RuntimeError("Cannot fit model before compiling. Call compile() first")

        for epoch in range(epochs):
            y_pred = self.forward(x_train)

            loss_gradient = self._loss.backward(y_pred, y_train)

            self.backward(loss_gradient)

            self._optimizer.step()
            self._optimizer.zero_grad()

            if callback(epoch, self):
                break

        return self
