from collections.abc import Callable, Sequence
from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from .module import Module
from .parameter import Parameter

__all__ = ["Sequential"]


def _noop_callback(_epoch: int, _model: Module) -> bool:
    """A no-operation callback function that does nothing."""
    return False


class Loss(Protocol):
    def forward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> np.float64: ...
    def backward(self, y_pred: NDArray[np.float64], y_true: NDArray[np.float64]) -> NDArray[np.float64]: ...


class Optimizer(Protocol):
    def step(self) -> None: ...
    def zero_grad(self) -> None: ...


class Sequential(Module):
    def __init__(self, layers: Sequence[Module]) -> None:
        self._layers = list(layers)

        self._loss: Loss | None = None
        self._optimizer: Optimizer | None = None

    def parameters(self) -> list[Parameter]:
        return [parameter for layer in self._layers for parameter in layer.parameters()]

    def forward(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        output = inputs

        for layer in self._layers:
            output = layer.forward(output)

        return output

    def backward(self, out_gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        gradient = out_gradient

        for layer in reversed(self._layers):
            gradient = layer.backward(gradient)

        return gradient

    def predict(self, inputs: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(inputs)

    def compile(self, *, optimizer: Optimizer, loss: Loss) -> None:
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
        if self._optimizer is None or self._loss is None:
            raise RuntimeError("Cannot fit model before compiling. Call compile() first")

        for epoch in range(epochs):
            y_pred = self.forward(x_train)

            loss_gradient = self._loss.backward(y_pred, y_train)

            self.backward(loss_gradient)

            self._optimizer.step()
            self._optimizer.zero_grad()

            early_stop = callback(epoch, self)

            if early_stop:
                break

        return self
