from typing import (
    Any,
    Dict,
    List
)

import pytest

from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import (
    LeaveOneOut,
    GridSearchCV
)

from corrosion.datasets import get_corrosion_dataset
from corrosion.enums import (
    Dataset,
    MetricType,
    TrainTestSplitType
)
from corrosion.metrics import Metrics
from corrosion.regressors import get_pca_ridge_model
from corrosion.train_test_split import TrainTestSplit

import deepchem as dc

# Constants
DATASET_PATH: str = get_corrosion_dataset(
    Dataset.Gas
)

BT_RATIO: float = 0.2
RANDOM_SEED: int = 12345
FINGERPRINT_SIZE: int = 32

PARAM_GRID: Dict[str, Any] = {
    "n_components": range(2, 8),
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100]
}


class TestDeepChemForCorrosionProject:

    def test_deepchem(self):

        # Load gas phase dataset
        data_gp_df: DataFrame = pd.read_csv(
            DATASET_PATH
        )

        # Create featurizer
        featurizer = dc.feat.CircularFingerprint(size=32)

        # Create fingerprints from SMILES
        fps = featurizer.featurize(data_gp_df.SMILES)

        # Add the fps to the dataset
        data_gp_df: DataFrame = pd.concat(
            (
                data_gp_df,
                pd.DataFrame(
                    fps,
                    columns=[f"FP_{_i + 1}" for _i in range(fps.shape[1])]  # 1 - n(features)
                )
            ),
            axis=1
        )

        # Get X
        _x_df: DataFrame = data_gp_df.iloc[:, 3:].drop(
            columns=["PAE", "EXP(PAE)", "Inhibitor Efficiency", "NEW Inhibitor Efficiency"],
            errors="ignore"
        )
        _x: ndarray = _x_df.values
        _x_labels: List[str] = list(_x_df.columns)

        # Get Y
        _y_theoretical: ndarray = data_gp_df["PAE"]

        # Split data
        _x_train, _x_bt, _y_theoretical_train, _y_theoretical_bt = TrainTestSplit.split(
            TrainTestSplitType.Random,
            _x,
            _y_theoretical,
            test_size=BT_RATIO,
            random_state=RANDOM_SEED
        )

        # Train Model
        _pca_ridge_theoretical: GridSearchCV = get_pca_ridge_model(
            param_grid=PARAM_GRID,
            cv=LeaveOneOut()
        )
        _pca_ridge_theoretical.fit(_x_train, _y_theoretical_train)

        # Calculate BT RMSE
        _bt_rmse: float = Metrics.get_metric(
            y=_y_theoretical_bt,
            y_hat=_pca_ridge_theoretical.predict(_x_bt),
            metric_type=MetricType.RMSE
        )

        print(_bt_rmse)


if __name__ == "__main__":
    pytest.run()
