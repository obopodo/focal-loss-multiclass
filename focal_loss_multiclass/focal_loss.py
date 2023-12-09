import lightgbm as lgb
import numpy as np

from .utils import one_hot_encode, softmax

# lgbm before 4.0.0 expects 1-D arrays
LGB_OLD = True if int(lgb.__version__.split(".")[0]) < 4 else False


class FocalLossMulti:
    def __init__(self, gamma: float, n_classes: int = 3, use_sk_api: bool = False) -> None:
        """Focal loss for multiclass classification.

        FL = -(1 - p_t)^gamma * log(p_t)

        The aim of this loss is to reduce the loss for well-classified examples
        and put more focus on hard, misclassified examples.

        Parameters
        ----------
        gamma : float
            Main parameter of focal loss. The higher gamma,
            the lower penalty for good enough predictions.
        n_classes : int, optional
            Number of classes, by default 3
        use_sk_api : bool, optional
            Whether to use scikit-learn API for LightGBM, by default False
        """
        self.gamma = gamma
        self.n_classes = n_classes
        self.use_sk_api = use_sk_api

    def _preds_for_true_class(self, y_true: np.ndarray, pred_probas: np.ndarray) -> np.ndarray:
        """Predicted probabilities for true class.

        Parameters
        ----------
        y_true : np.ndarray (n_samples,)
            Class labels
        pred_probas : np.ndarray (n_samples, n_classes)
            Probabilities for each class

        Returns
        -------
        np.ndarray (n_samples,)
            Probabilities for true class
        """
        return pred_probas[np.arange(len(pred_probas)), y_true.astype(int)]  # preds for true class

    def __call__(self, y_true: np.ndarray, pred_probas: np.ndarray) -> np.ndarray:
        """Focal loss for the whole dataset.

        Parameters
        ----------
        y_true : np.ndarray (n_samples,)
            Class labels
        pred_probas : np.ndarray (n_samples, n_classes)
            Predicted probabilities for each class

        Returns
        -------
        np.ndarray (n_samples,)
            Focal loss value for each observation
        """
        p_true_class = self._preds_for_true_class(y_true, pred_probas)
        return -((1 - p_true_class) ** self.gamma) * np.log(p_true_class)

    def grad_hess(
        self, y_true: np.ndarray, pred_probas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gradient and Hessian of focal loss for the whole dataset.

        Each row of grad and hess corresponds to a single observation.
        Each column corresponds to a single class.


        Parameters
        ----------
        y_true : np.ndarray (n_samples,)
            Class labels
        pred_probas : np.ndarray (n_samples, n_classes)
            Class probabilities - predictions after sigmoid

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Gradient and Hessian of focal loss.
            Each array has shape (n_samples, n_classes)
        """

        y_ohe = one_hot_encode(y_true, self.n_classes)
        p_true_class = self._preds_for_true_class(y_true, pred_probas)

        class_factor = y_ohe - pred_probas

        log_p_t = np.log(p_true_class)
        p_1 = 1 - p_true_class
        p_1_g_2 = p_1 ** (self.gamma - 2)

        # Gradient
        G_AA = self.gamma * p_true_class * (p_1 ** (self.gamma - 1)) * log_p_t
        G_BB = p_1**self.gamma
        grad = (G_AA - G_BB).reshape(-1, 1) * class_factor

        # Hessian
        H_A1 = self.gamma * p_true_class * p_1_g_2
        H_A2 = (p_1) * (log_p_t + 2) - (self.gamma - 1) * p_true_class
        H_AA = (H_A1 * H_A2).reshape(-1, 1) * class_factor

        H_B1 = p_1_g_2
        H_B2 = self.gamma * p_true_class * (p_1) * log_p_t
        H_B3 = (p_1) ** 2
        H_BB = (H_B1 * (H_B2 - H_B3)).reshape(-1, 1) * pred_probas * (1 - pred_probas)

        hess = H_AA - H_BB

        return grad, 2 * hess

    def lgb_obj(self, preds: np.ndarray, train_data: lgb.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """LightGBM objective function for focal loss.

        Parameters
        ----------
        preds : np.ndarray
            Raw preds from LightGBM, float values in (-inf, inf)
        train_data : lgb.Dataset
            Train dataset, used to get class labels

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            grad and hess of focal loss
        """
        y_true = train_data.get_label()
        if LGB_OLD:
            # lgbm before 4.0.0 returns preds as 1-D array
            preds = preds.reshape((y_true.size, -1), order="F")

        pred_probas = softmax(preds)
        grad, hess = self.grad_hess(y_true=y_true, pred_probas=pred_probas)

        if LGB_OLD:
            # lgbm before 4.0.0 expects 1-D arrays
            grad = grad.ravel(order="F")
            hess = hess.ravel(order="F")
        return grad, hess

    def lgb_eval(self, preds: np.ndarray, train_data: lgb.Dataset) -> tuple[str, float, bool]:
        """LightGBM evaluation function for focal loss.

        Parameters
        ----------
        preds : np.ndarray
            Raw preds from LightGBM, float values in (-inf, inf)
        train_data : lgb.Dataset
            Train dataset, used to get class labels

        Returns
        -------
        tuple[str, float, bool]
            Tuple with metric name, metric value, is_higher_better
        """
        y_true = train_data.get_label()
        if LGB_OLD:
            # lgbm before 4.0.0 returns preds as 1-D array
            preds = preds.reshape((y_true.size, -1), order="F")

        pred_probas = softmax(preds)
        is_higher_better = False

        return "focal_loss", self(y_true, pred_probas).mean(), is_higher_better
