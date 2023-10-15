from typing import (
    Any,
    Dict,
    Optional
)

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


class PcaRidge(BaseEstimator):

    def __init__(
        self,
        n_components: int = 10,
        alpha: float = 0.001,
        b_scale_data: bool = False
    ):

        self.n_components = n_components
        self.alpha = alpha
        self.b_scale_data = b_scale_data

        self.__model: Optional[Pipeline] = None

    def __get_model(self):

        _pipeline_elements = []

        if self.b_scale_data:
            _pipeline_elements.append(("StandardScaler", StandardScaler()))

        _pipeline_elements.append(("PCA", PCA(n_components=self.n_components)))
        _pipeline_elements.append(("Ridge", Ridge(alpha=self.alpha)))

        return Pipeline(_pipeline_elements)

    def get_params(self, deep=True):
        return {
            "n_components": self.n_components,
            "alpha": self.alpha
        }

    def fit(self, X, y):  # noqa (uppercase X)
        self.__model = self.__get_model()
        self.__model.fit(X, y)
        return self

    def predict(self, X):  # noqa (uppercase X)
        return self.__model.predict(X)


def get_pca_ridge_model(
    param_grid: Dict[str, Any],
    cv: Any,
    b_scale_data: bool = False,   # Usually no scaling for (somewhat) independent parameters
    scoring: str = "neg_root_mean_squared_error",
    verbose: int = 2,
    n_jobs: int = 1
):
    _pca_ridge: PcaRidge = PcaRidge(
        b_scale_data=b_scale_data
    )

    _model: GridSearchCV = GridSearchCV(
        estimator=_pca_ridge,
        param_grid=param_grid,
        scoring=scoring,
        verbose=verbose,
        n_jobs=n_jobs,
        cv=cv
    )

    return _model
