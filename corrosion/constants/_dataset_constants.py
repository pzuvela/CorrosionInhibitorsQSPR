import os


class DatasetConstants:

    DATA: str = "data"
    DATASET_NAME_GP: str = "2023-10-13-corrosion_inhibition_dataset_gp.csv"
    DATASET_NAME_SP: str = "2023-10-10-corrosion_inhibition_dataset_sp.csv"

    MAIN_PATH: str = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    DATA_PATH: str = os.path.join(MAIN_PATH, DATA)

