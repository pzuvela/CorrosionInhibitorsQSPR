from typing import (
    Callable,
    Dict
)

import numpy as np
from numpy import ndarray

from corrosion.enums import MetricType
from corrosion.exceptions import Exceptions


class Metrics:

    @staticmethod
    def get_metric(
        y: ndarray,
        y_hat: ndarray,
        metric_type: MetricType
    ) -> float:
        _metrics_mapping: Dict[MetricType, Callable] = {
            MetricType.R2: Metrics.__get_r2,
            MetricType.MSE: Metrics.__get_mse,
            MetricType.RMSE: Metrics.__get_rmse,
            MetricType.RE: Metrics.__get_re
        }
        return _metrics_mapping.get(metric_type, Exceptions.invalid_metric_type)(
            y,
            y_hat
        )

    @staticmethod
    def __get_r2(
        y: ndarray,
        y_hat: ndarray
    ) -> float:
        _ss_res: float = np.sum((y.ravel() - y_hat.ravel()) ** 2).item()
        _ss_tot: float = np.sum((y.ravel() - np.mean(y.ravel())) ** 2).item()
        return 1 - (_ss_res / _ss_tot)

    @staticmethod
    def __get_mse(
        y: ndarray,
        y_hat: ndarray
    ) -> float:
        return np.mean(
            (y_hat.ravel() - y.ravel()) ** 2
        ).item()

    @staticmethod
    def __get_rmse(
        y: ndarray,
        y_hat: ndarray
    ) -> float:
        return np.sqrt(
            Metrics.__get_mse(
                y.ravel(),
                y_hat.ravel()
            )
        )

    @staticmethod
    def __get_re(
        y: ndarray,
        y_hat: ndarray
    ) -> float:
        return (abs(y_hat.ravel() - y.ravel()) / y.ravel()) * 100
