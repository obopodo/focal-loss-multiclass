import numpy as np


def softmax(x: np.ndarray, axis=1) -> np.ndarray:
    """Computationaly Stable Softmax.

    from https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/

    Parameters
    ----------
    x : np.ndarray
        Input array. In multiclass case should be 2D array (n_samples, n_classes)
    axis : int, optional
        Axis to apply softmax, by default 1

    Returns
    -------
    np.ndarray
        Softmaxed array
    """
    y = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(y) / np.sum(np.exp(y), axis=axis, keepdims=True)


def one_hot_encode(y_true: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Naive one-hot-encoding.

    Parameters
    ----------
    y_true : np.ndarray (n_samples,)
        Class labels
    n_classes : int, optional
        Number of classes, by default None
        If None, will be inferred from y_true

    Returns
    -------
    np.ndarray (n_samples, n_classes)
        One-hot-encoded labels
    """
    if n_classes is None:
        n_classes = np.unique(y_true).shape[0]

    # Create an identity matrix of size n_classes
    identity = np.eye(n_classes)

    # Use y_true as indices to select rows from the identity matrix
    y_true_one_hot = identity[y_true.astype(int)]
    return y_true_one_hot
