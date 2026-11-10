"""ADALINE (Adaptive Linear Neuron) model implementation.

This implementation is based on the original ADALINE algorithm introduced by Bernard Widrow and Marcian Hoff.
"""

from collections.abc import Callable
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["ADALINE"]


def _noop_callback(_epoch: int, _model: Any) -> bool:
    """A no-operation callback function that does nothing."""
    return False


class ADALINE:
    """ADALINE (Adaptive Linear Neuron) model algorithm.

    Args:
        in_features: Number of input features for the model.

    Example:
        >>> import numpy as np
        >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        >>> y = np.array([0, 0, 0, 1], dtype=np.float64)
        >>> model = ADALINE(in_features=2)
        >>> model.fit(X, y, learning_rate=0.02, epochs=100)
        >>> y_pred = model.predict(np.array([1, 1], dtype=np.float64))
        >>> np.where(y_pred >= 0.5, 1.0, 0.0)  # Convert to binary output
        1.0
    """

    def __init__(self, in_features: int) -> None:
        self._weights = np.zeros(in_features, dtype=np.float64)
        self._bias = np.float64(0.0)

    @property
    def weights(self) -> NDArray[np.float64]:
        """ADALINE model weights."""
        return self._weights

    @property
    def bias(self) -> np.float64:
        """ADALINE model bias."""
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
        """Applies the identity activation function (linear activation).

        Args:
            x: Input value.

        Returns:
            The same input value.
        """
        return x

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
        learning_rate: float = 0.01,
        epochs: int = 100,
        callback: Callable[[int, Self], bool] = _noop_callback,
    ) -> Self:
        """Fits the ADALINE model using stochastic gradient descent (SGD).

        Args:
            X_train: Training samples.
            y_train: Target values.
            learning_rate: Scaling factor for weight updates.
            epochs: Maximum number of passes over the training dataset.
            callback: Function called at the end of each epoch.
                      Returns True to stop training early.

        Returns:
            Trained ADALINE instance.
        """
        for epoch in range(epochs):
            for features, y_true in zip(X_train, y_train):
                # Predict the output for the current features
                y_pred = self.predict(features)

                # Compute the prediction error (difference between true and predicted values)
                error = y_true - y_pred

                # Update weights and bias using stochastic gradient descent (SGD)
                self._weights += learning_rate * error * features
                self._bias += learning_rate * error

            early_stop = callback(epoch, self)

            # There is no convergence check, as it uses a continuous linear error
            # to minimize a quadratic loss

            if early_stop:  # Stop early if the callback returns True
                break

        return self

    def __call__(self, features: NDArray[np.float64]) -> np.float64:
        """Perform a prediction for a given input vector."""
        return self.predict(features)

    def __repr__(self) -> str:
        """Returns a string representation of the ADALINE model."""
        return f"ADALINE(weights={self._weights}, bias={self._bias})"
