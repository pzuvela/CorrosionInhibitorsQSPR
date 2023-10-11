from typing import Optional

import numpy as np
from numpy import ndarray


class ApplicabilityDomain:

    @staticmethod
    def __get_core(
        x: ndarray
    ) -> ndarray:
        return np.linalg.pinv(x.transpose() @ x)

    @staticmethod
    def __get_leverage(
        x: ndarray,
        h_core: ndarray
    ) -> ndarray:
        return np.diag((x @ h_core) @ x.transpose())

    @staticmethod
    def calculate(
        x_train: ndarray,
        y_train: ndarray,
        y_train_hat: ndarray,
        x_validation: Optional[ndarray] = None,
        y_validation: Optional[ndarray] = None,
        y_validation_hat: Optional[ndarray] = None,
        x_bt: Optional[ndarray] = None,
        y_bt: Optional[ndarray] = None,
        y_bt_hat: Optional[ndarray] = None
    ):

        b_has_validation = x_validation is not None and y_validation is not None and y_validation_hat is not None
        b_has_bt = x_bt is not None and y_bt is not None and y_bt_hat is not None

        k: int = x_train.shape[1] + 1
        n: int = x_train.shape[0]

        hat_star: float = 3 * (k / n)

        _mean_x = np.mean(x_train)

        x_train -= _mean_x
        if b_has_validation:
            x_validation -= _mean_x
        if b_has_bt:
            x_bt -= _mean_x

        h_core: ndarray = ApplicabilityDomain.__get_core(x_train)

        hat_train: ndarray = ApplicabilityDomain.__get_leverage(x_train, h_core)
        res_train: ndarray = y_train_hat - y_train
        s: float = float(np.std(res_train))
        res_scaled_train: ndarray = res_train / s

        hat_validation: Optional[ndarray] = None
        res_scaled_validation: Optional[ndarray] = None

        if b_has_validation:
            hat_validation: ndarray = ApplicabilityDomain.__get_leverage(x_validation, h_core)
            res_validation: ndarray = y_validation_hat - y_validation
            res_scaled_validation: ndarray = res_validation / s

        hat_bt: Optional[ndarray] = None
        res_scaled_bt: Optional[ndarray] = None

        if b_has_bt:
            hat_bt: ndarray = ApplicabilityDomain.__get_leverage(x_bt, h_core)
            res_bt: ndarray = y_bt_hat - y_bt
            res_scaled_bt: ndarray = res_bt / s

        return (
            hat_star,
            hat_train, hat_validation, hat_bt,
            res_scaled_train, res_scaled_validation, res_scaled_bt
        )
