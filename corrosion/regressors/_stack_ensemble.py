from typing import (
    Any,
    Tuple
)

import numpy as np
from numpy import ndarray

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression


class StackingRegressor(BaseEstimator):

    def __init__(self, *models):
        self.models: Tuple[Any] = models
        self.__stack_model: LinearRegression = LinearRegression()
        self.IS_FITTED: bool = False

    def __initial_predict(self, X) -> ndarray:  # noqa (uppercase X)
        return np.hstack(
            [_model.predict(X).reshape(-1, 1) for _model in self.models]
        )

    def fit(self, X, y):  # noqa (uppercase X)
        _x = self.__initial_predict(X)
        self.__stack_model.fit(_x, y)
        self.IS_FITTED = True

    def predict(self, X) -> ndarray:  # noqa (uppercase X)
        _x = self.__initial_predict(X)
        return self.__stack_model.predict(_x).reshape(-1, 1)
