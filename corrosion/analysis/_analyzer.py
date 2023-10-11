import numpy as np
from pandas import DataFrame


class Analyzer:

    @staticmethod
    def get_optimal_n_pcs(
        explained_variance_df: DataFrame,
        threshold: float = 0.01,
        explained_variance_label: str = "%Explained Variance"
    ) -> int:

        # Calculate the gradient of the cumulative explained variance
        gradient = np.gradient(explained_variance_df[explained_variance_label])

        # Find the index of the knee point
        n_components = np.where(gradient >= threshold)[-1][-1] + 1  # Last true occurrence +1

        return n_components
