from enum import Enum


class FeatureImportanceType(Enum):
    MeanImpurityDecrease = 1
    FeaturePermutation = 2
    SHAP = 3
