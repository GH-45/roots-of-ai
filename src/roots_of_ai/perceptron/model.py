"""Simple Perceptron model implementation.

This implementation is based on the original Perceptron algorithm introduced by Frank Rosenblatt.
"""

from collections.abc import Callable
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["Perceptron"]


def _noop_callback(_epoch: int, _model: Any) -> None:
    """A no-operation callback function that does nothing."""


class Perceptron:
    """Perceptron model algorithm.

    Args:
        in_features: Number of input features for the model.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        >>> y = np.array([0, 0, 0, 1], dtype=np.float64)
        >>> model = Perceptron(in_features=2)
        >>> model.fit(X, y, learning_rate=0.5, epochs=10)
        >>> model.predict(np.array([1, 1], dtype=np.float64))
        1.0
    """

    def __init__(self, in_features: int) -> None:
        self._weights = np.zeros(in_features, dtype=np.float64)
        self._bias = np.float64(0.0)

    @property
    def weights(self) -> NDArray[np.float64]:
        """Perceptron model weights."""
        return self._weights

    @property
    def bias(self) -> np.float64:
        """Perceptron model bias."""
        return self._bias

    def _weighted_sum(self, features: NDArray[np.float64]) -> np.float64:
        """Computes the weighted sum of the input features plus the bias.

        Args:
            features: Input feature vector.

        Returns:
            Result of the weighted sum.
        """
        return np.float64(features @ self._weights + self._bias)

    def _activation(self, x: np.float64) -> np.float64:
        """Applies the Heaviside step activation function.

        Args:
            x: Input value.

        Returns:
            One if the input is greater than or equal to zero, otherwise zero.
        """
        return np.float64(1.0 if x >= 0.0 else 0.0)

    def predict(self, features: NDArray[np.float64]) -> np.float64:
        """Perform a prediction for a given input vector.

        Args:
            features: Input feature vector.

        Returns:
            Model prediction.
        """
        return self._activation(self._weighted_sum(features))

    def fit(
        self,
        X_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        learning_rate: float = 0.5,
        epochs: int = 100,
        callback: Callable[[int, Self], None] = _noop_callback,
    ) -> Self:
        """Fits the Perceptron model using the convergence rule.

        Args:
            X_train: Training samples.
            y_train: Target values.
            learning_rate: Scaling factor for weight updates.
            epochs: Maximum number of passes over the training dataset.
            callback: Function called at the end of each epoch.

        Returns:
            Trained Perceptron instance.
        """
        for epoch in range(epochs):
            converged = True

            for features, y_true in zip(X_train, y_train):
                # Predict the output for the current features
                y_pred = self.predict(features)

                # Compute the prediction error (difference between true and predicted values)
                error = y_true - y_pred

                # Update weights and bias if a misclassification occurs
                if error != 0:
                    # Update weights and bias using the Perceptron convergence rule
                    self._weights += learning_rate * error * features
                    self._bias += learning_rate * error

                    converged = False

            callback(epoch, self)

            if converged:  # Stop early if the model has converged
                break

        return self

    def __call__(self, features: NDArray[np.float64]) -> np.float64:
        """Perform a prediction for a given input vector."""
        return self.predict(features)

    def __repr__(self) -> str:
        """Returns a string representation of the Perceptron model."""
        return f"SimplePerceptron(weights={self._weights}, bias={self._bias})"
