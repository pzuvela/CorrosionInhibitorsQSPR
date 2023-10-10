from typing import (
    Any,
    Tuple
)

import numpy as np
from numpy import ndarray


class EnsembleRegressor:

    def __init__(self, *models):
        self.models: Tuple[Any] = models

    def __initial_predict(self, X) -> ndarray:  # noqa (uppercase X)
        return np.hstack(
            [_model.predict(X).reshape(-1, 1) for _model in self.models]
        )

    def predict(self, X) -> ndarray:  # noqa (uppercase X)
        _x = self.__initial_predict(X)
        return np.mean(_x, axis=1).reshape(-1, 1)
