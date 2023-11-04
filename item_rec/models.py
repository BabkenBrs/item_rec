"""This module saves different models."""

import implicit
import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp
from tqdm import tqdm


class RandomModel:
    """Random model class.

    This model randomly choose new never-seen items for user.
    """

    def fit(self, train_dataset: sp.csr_matrix) -> None:
        """Train model.

        Args:
            train_dataset: dataset wihtout test items
        """
        self.train_dataset = train_dataset.toarray()
        self.num_of_users = train_dataset.shape[0]
        self.recommends: list[NDArray] = [None for i in range(self.num_of_users)]  # type: ignore

        for user in tqdm(range(self.num_of_users)):
            self.recommends[user] = np.random.choice(  # type: ignore
                np.where(self.train_dataset[user] != 1.0)[0], 10
            )

    def predict(self) -> list[NDArray]:
        """This function gives recommendations for users.

        Returns:
            recommends: list of recommendations for users
        """
        return self.recommends


class IALS:
    """Implicit Alternating Least Squares model.

    Attributes:
        model: trained model that makes recommendations
    """

    def __init__(
        self, emb_size: int = 100, reg_coef: float = 0.1, n_iter: int = 15
    ) -> None:
        """Initializes the instance of IALS model.

        Args:
            emb_size: size of user and item embeddings
            reg_coef: regularization coefficient
            n_iter: number of iterations
        """
        self.model = implicit.als.AlternatingLeastSquares(
            factors=emb_size, regularization=reg_coef, iterations=n_iter
        )

    def fit(
        self, train_dataset: sp.csr_matrix
    ) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Train model.

        Args:
            train_dataset: dataset wihtout test items
        """
        self.train_dataset = train_dataset
        self.model.fit(train_dataset)

        return self.model.user_factors, self.model.item_factors

    def predict(self) -> list[NDArray[np.int_]]:
        """Predict recommendations for users.

        Returns:
            recommends: list of recommendations for users
        """
        recommends: list[NDArray] = [None for i in range(self.train_dataset.shape[0])]  # type: ignore
        for user in range(self.train_dataset.shape[0]):
            recommends[user] = self.model.recommend(
                user, self.train_dataset[user], filter_already_liked_items=True
            )[0]

        return recommends
