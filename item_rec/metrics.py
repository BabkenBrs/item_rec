"""Module for metrics."""

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


class Map10:
    """Class which help to count Mean Average Precision at 10.

    Attributes:
        recommendations: user recommendations
        real_test: real user preferences
    """

    def __init__(
        self,
        recommendations: list[NDArray[np.int_]],
        real_test: dict[int, NDArray[np.int_]],
    ) -> None:
        """Initialiazes the instance for calcuflating Mean Average Precision at 10.

        Args:
            recommendations: A list which contain recommendations for users
            real_test: A dictionary mapping users to items with which the user interacted
        """
        self.recommendations = recommendations
        self.real_test = real_test

    def precision_k(
        self, k: int, user_recommend: NDArray[np.int_], user_real: NDArray[np.int_]
    ) -> float:
        """Calculate precision at k first recommendations.

        Args:
            k: size of recommendation list
            user_recommend: list of sorted recommendations for user
            user_real: list of real items with which the user interacted

        Returns:
            Precision at first k recommendations. For example:

            0.2

            Returned precision is always float
        """
        rel = len(np.intersect1d(user_recommend[:k], user_real))
        return rel / k

    def average_precision_k(
        self, K: int, user_recommend: NDArray[np.int_], user_real: NDArray[np.int_]
    ) -> float:
        """Calculate average precision at K first recommendations.

        Args:
            K: number of top recommendations
            user_recommend: list of sorted recommendations for user
            user_real: list of real items with which the user interacted

        Returns:
            Average Precision at first K recommendations. For example:

            0.456
        """
        cnt = 0
        sum_prec_k = 0.0
        for k in range(1, K + 1):
            if user_recommend[k - 1] in user_real:
                cnt += 1
                sum_prec_k += self.precision_k(k, user_recommend[:k], user_real)
        return sum_prec_k / max(cnt, 1.0)

    def calculate_map_10(self):
        """Calculate Mean Average Precision at 10."""
        return np.mean(
            [
                self.average_precision_k(
                    10, self.recommendations[user], self.real_test[user]
                )
                for user in tqdm(range(len(self.recommendations)))
            ]
        )
