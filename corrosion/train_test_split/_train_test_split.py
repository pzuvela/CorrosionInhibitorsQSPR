from typing import (
    Dict,
    Callable
)

import numpy as np
from numpy import ndarray

from sklearn.model_selection import train_test_split

from corrosion.enums import TrainTestSplitType


class TrainTestSplit:

    @staticmethod
    def split(
        train_test_split_type: TrainTestSplitType,
        *args,
        **kwargs
    ):

        _MAPPING: Dict[TrainTestSplitType, Callable] = {
            TrainTestSplitType.Kenstone: TrainTestSplit.kenstone,
            TrainTestSplitType.Random: train_test_split
        }

        return _MAPPING[train_test_split_type](
            *args,
            **kwargs
        )

    @staticmethod
    def kenstone(x: ndarray, num_test: int):

        original_x = x

        distance_to_average = ((x - np.tile(x.mean(axis=0), (x.shape[0], 1))) ** 2).sum(axis=1)
        max_dist_num = np.where(distance_to_average == np.max(distance_to_average))
        max_dist_num = max_dist_num[0][0]

        sel_nums = list()
        sel_nums.append(max_dist_num)
        remain_nums = np.arange(0, x.shape[0], 1)

        x = np.delete(x, sel_nums, 0)

        remain_nums = np.delete(remain_nums, sel_nums, 0)

        for iteration in range(1, num_test):
            sel_obs = original_x[sel_nums, :]
            min_dist_to_sel_obs = list()
            for min_dist_num in range(0, x.shape[0]):
                distance_to_sel_obs = (
                        (sel_obs - np.tile(x[min_dist_num, :], (sel_obs.shape[0], 1))) ** 2
                ).sum(axis=1)
                min_dist_to_sel_obs.append(np.min(distance_to_sel_obs))
            max_dist_num = np.where(min_dist_to_sel_obs == np.max(min_dist_to_sel_obs))
            max_dist_num = max_dist_num[0][0]
            sel_nums.append(remain_nums[max_dist_num])
            x = np.delete(x, max_dist_num, 0)
            remain_nums = np.delete(remain_nums, max_dist_num, 0)

        return remain_nums.tolist(), sel_nums
