import os
import sys
from typing import (
    Any,
    Dict,
    List
)

from numpy import ndarray

import pandas as pd
from pandas import DataFrame

from sklearn.model_selection import (
    LeaveOneOut,
    GridSearchCV
)


# Constants

DATA: str = "data"
DATASET_NAME: str = "2023-10-13-corrosion_inhibition_dataset_gp.csv"
MAIN_PATH: str = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))

if MAIN_PATH not in sys.path:
    sys.path.insert(0, MAIN_PATH)

DATA_PATH: str = os.path.join(MAIN_PATH, DATA)
DATASET_PATH: str = os.path.join(DATA_PATH, DATASET_NAME)

BT_RATIO: float = 0.2
RANDOM_SEED: int = 12345

PARAM_GRID: Dict[str, Any] = {
    "n_components": range(2, 8),
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 50, 100]
}

from corrosion.enums import TrainTestSplitType  # noqa (import after placing corrosion on the path)
from corrosion.regressors import get_pca_ridge_model  # noqa (import after placing corrosion on the path)
from corrosion.train_test_split import TrainTestSplit  # noqa (import after placing corrosion on the path)


if __name__ == "__main__":

    # Load Data
    data_df: DataFrame = pd.read_csv(DATASET_PATH)

    # Get only quinolines
    data_quinolines_df: DataFrame = data_df[data_df.Name.str.contains("quinoline")].reset_index(drop=True)

    # Get X
    _x_df: DataFrame = data_quinolines_df.iloc[:, 3:].drop(
        columns=["PAE", "EXP(PAE)", "Inhibitor Efficiency", "NEW Inhibitor Efficiency"],
        errors="ignore"
    )
    _x: ndarray = _x_df.values
    _x_labels: List[str] = list(_x_df.columns)

    # Number of compounds & features
    m: int
    n: int
    m, n = _x.shape

    # Get Y
    _y_theoretical: ndarray = data_quinolines_df["PAE"]

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
