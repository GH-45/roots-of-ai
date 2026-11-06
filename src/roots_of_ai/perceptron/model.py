from collections.abc import Callable
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["Perceptron"]


def _noop_callback(_epoch: int, _model: Any) -> None:
    pass


class Perceptron:
    def __init__(self, in_features: int) -> None:
        self._weights = np.zeros(in_features, dtype=np.float64)
        self._bias = np.float64(0.0)

    def _weighted_sum(self, features: NDArray[np.float64]) -> np.float64:
        return np.float64(features @ self._weights + self._bias)

    def _activation(self, x: np.float64) -> np.float64:
        return np.float64(1.0 if x >= 0.0 else 0.0)

    def predict(self, features: NDArray[np.float64]) -> np.float64:
        return self._activation(self._weighted_sum(features))

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        learning_rate: float = 0.5,
        epochs: int = 100,
        callback: Callable[[int, Self], Any] = _noop_callback,
    ) -> Self:
        for epoch in range(epochs):
            converged = True

            for features, y_true in zip(X_train, y_train):
                # Predict the output for the current features
                y_pred = self.predict(features)

                # Calculate the error (difference between true and predicted values)
                error = y_true - y_pred

                # Update weights and bias if there is an error
                if error != 0:
                    # Update weights and bias using the perceptron convergence rule (Rosenblatt's rule)
                    self._weights += learning_rate * error * features
                    self._bias += learning_rate * error

                    converged = False

            callback(epoch, self)

            if converged:  # If no errors were found in this epoch, the model has converged
                break

        return self

    def __call__(self, features: NDArray[np.float64]) -> np.float64:
        return self.predict(features)

    def __repr__(self) -> str:
        return f"SimplePerceptron(weights={self._weights}, bias={self._bias})"
