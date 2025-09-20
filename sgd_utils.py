import numpy as np


def _add_bias(X: np.ndarray) -> np.ndarray:
    """Return X with a leading column of ones for the bias term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def stochastic_gradient_descent(
    X,
    y,
    learning_rate: float = 0.01,
    epochs: int = 50,
    batch_size: int = 256,
    random_state: int | None = 42,
    l2: float = 0.0,
):
    """Train linear regression parameters using mini-batch SGD.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target values.
    learning_rate : float, default=0.01
        Step size for the gradient update.
    epochs : int, default=50
        Number of passes over the training set.
    batch_size : int, default=256
        Size of the mini-batches. Values <= 0 fall back to full batch.
    random_state : int or None, default=42
        Seed used to shuffle the data before each epoch.
    l2 : float, default=0.0
        L2 regularisation strength. Use 0.0 to disable.

    Returns
    -------
    theta : ndarray of shape (n_features + 1,)
        Estimated parameters including the bias term.
    history : list[float]
        Mean squared error after each epoch.
    """
    rng = np.random.default_rng(random_state)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if y.ndim != 1:
        y = y.ravel()

    n_samples, n_features = X.shape
    batch_size = int(batch_size) if batch_size else n_samples
    batch_size = max(1, min(batch_size, n_samples))

    X_bias = _add_bias(X)
    theta = rng.normal(loc=0.0, scale=0.01, size=n_features + 1)

    history = []
    for _ in range(int(epochs)):
        indices = rng.permutation(n_samples)
        X_shuffled = X_bias[indices]
        y_shuffled = y[indices]

        for start in range(0, n_samples, batch_size):
            xb = X_shuffled[start : start + batch_size]
            yb = y_shuffled[start : start + batch_size]

            errors = xb @ theta - yb
            grad = xb.T @ errors / len(xb)
            if l2:
                grad[1:] += l2 * theta[1:]

            theta -= learning_rate * grad

        mse = np.mean((X_bias @ theta - y) ** 2)
        history.append(float(mse))

    return theta, history


def predict_with_theta(X, theta):
    """Predict targets for X given parameters theta returned by SGD."""
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim != 1:
        theta = theta.ravel()
    return _add_bias(X) @ theta
