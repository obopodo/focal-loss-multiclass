import numpy as np
import pytest

from focal_loss_multiclass import FocalLossMulti, one_hot_encode, softmax


@pytest.fixture(scope="module")
def dataset() -> tuple[np.ndarray, np.ndarray]:
    """Create a dataset for testing.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple with two arrays:
        y_true (n_samples,)
        preds_proba (n_samples, n_classes)
    """
    n_samples = 100
    n_classes = 3
    np.random.seed(73)
    y_true = np.random.randint(0, n_classes, size=n_samples)
    preds_raw = np.random.rand(n_samples, n_classes)
    preds_proba = softmax(preds_raw, axis=1)
    return y_true, preds_proba


def logloss(y_true: np.ndarray, preds_proba: np.ndarray) -> np.ndarray:
    """Calculate the logloss for the dataset

    Parameters
    ----------
    y_true: np.ndarray (n_samples,)
        Class labels
    preds_proba: np.ndarray (n_samples, n_classes)
        Predicted probabilities for each class

    Returns
    -------
    np.ndarray (n_samples,)
        Logloss for each observation
    """
    return -np.log(preds_proba[np.arange(len(preds_proba)), y_true.astype(int)])


def logloss_grad_hess(y_true: np.ndarray, preds_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logloss gradient and hessian.

    Parameters
    ----------
    y_true: np.ndarray (n_samples,)
        Class labels
    preds_proba: np.ndarray (n_samples, n_classes)
        Predicted probabilities for each class

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        gradient (n_samples, n_classes)
        hessian (n_samples, n_classes)
    """
    y_true_ohe = one_hot_encode(y_true)
    grad = preds_proba - y_true_ohe
    hess = preds_proba * (1 - preds_proba)
    return grad, 2 * hess


def test_gamma_0(dataset):
    """Test case where gamma=0.

    In this case the focal loss should be equal to the logloss.

    Parameters
    ----------
    dataset : tuple[np.ndarray, np.ndarray]
        Tuple with two arrays:
        y_true: np.ndarray (n_samples,)
        preds_proba: np.ndarray (n_samples, n_classes)
    """
    y_true, preds_proba = dataset
    fl = FocalLossMulti(gamma=0)

    loss = fl(y_true, preds_proba)
    assert loss.shape == y_true.shape
    assert np.allclose(loss, logloss(y_true, preds_proba)), "Loss is not correct"

    grad, hess = fl.grad_hess(y_true, preds_proba)
    grad_ll, hess_ll = logloss_grad_hess(y_true, preds_proba)
    assert grad.shape == grad_ll.shape
    assert hess.shape == hess_ll.shape
    assert np.allclose(grad, grad_ll), "Gradient is not correct"
    assert np.allclose(hess, hess_ll), "Hessian is not correct"
