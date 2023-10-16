import os
from typing import Dict

from corrosion.constants import DatasetConstants
from corrosion.enums import Dataset


def get_corrosion_dataset(
    dataset: Dataset
) -> str:

    _MAPPING: Dict[Dataset, str] = {
        Dataset.Gas: DatasetConstants.DATASET_NAME_GP,
        Dataset.Solvent: DatasetConstants.DATASET_NAME_SP
    }

    return os.path.join(
        DatasetConstants.DATA_PATH,
        _MAPPING[dataset]
    )
