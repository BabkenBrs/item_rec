""""Module for getting and preprocessing data."""

import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy import sparse as sp


class PreprocessData:
    """Class for preprocessing data.

    Attributes:
        path_to_file: path to the dataset
    """

    def __init__(self, path_to_file: str) -> None:
        """Initialiazes the instance for data preprocessing.

        Args:
            path_to_file: path to the file, where we store data
        """
        self.path_to_data = path_to_file

    def get_data(self) -> sp.csr_matrix:
        """Loading dataset and convert it to sparse matrix format.

        Returns:
            Dataset of format scipy.sparse.csr_matrix
        """
        interactions = pd.read_csv(self.path_to_data)

        # self.set_of_users = interactions.row.unique()
        # self.set_of_items = interactions.col.unique()

        # sorted_set_of_users = sorted(self.set_of_users)
        # sorted_set_of_items = sorted(self.set_of_items)

        # self.raw_to_usr = {raw: usr for raw, usr in enumerate(sorted_set_of_users)}
        # self.col_to_item = {col: item for col, item in enumerate(sorted_set_of_items)}

        self.new_inter = pd.pivot_table(
            interactions, index="row", columns="col", values="data"
        ).fillna(0)

        self.dataset = sp.csr_matrix(self.new_inter)

        return self.dataset

    def train_test_split(
        self, dataset: sp.csr_matrix, train_size: float = 0.8
    ) -> tuple[sp.csr_matrix, dict[int, NDArray[np.int_]]]:
        """Split our data to train and test parts.

        Args:
            dataset: sparse matrix of user-item interaction
            train_size: share of training part in dataset

        Returns:
            Tuple with 2 objects in it:
            1) training dataset
            2) test part of user item interactions.

            For example:

            (
                np.array(
                    [
                        [0., 0., 1., 0.],
                        [1., 0., 0., 1.]
                    ]
                ),
                {
                    0 : np.array([3]),
                    1 : np.array([2])
                }
            )
        """
        self.train_dataset = dataset.toarray()

        self.train_usr_item = {}
        self.test_usr_item = {}
        for user in range(self.train_dataset.shape[0]):
            user_items = np.where(self.train_dataset[user] == 1.0)[0]
            if len(user_items) == 1:
                self.train_usr_item[user] = []
                self.test_usr_item[user] = user_items
            else:
                num_of_items = len(user_items)
                self.train_usr_item[user] = np.random.choice(
                    user_items, int(train_size * num_of_items), replace=False
                )
                self.test_usr_item[user] = np.setdiff1d(
                    user_items, self.train_usr_item[user]
                )

                self.train_dataset[user][self.test_usr_item[user]] = 0.0

        self.train_dataset = sp.csr_matrix(self.train_dataset)

        return self.train_dataset, self.test_usr_item
