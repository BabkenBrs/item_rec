"""This module is for different helping functions and classes."""
import os

from datasets import load_dataset


def load_data():
    """Function for loading data."""
    if not os.path.exists("data"):
        os.mkdir("data")
        data_files = {
            "interactions": "interactions.csv",
            "item_price": "item_price.csv",
            "item_asset": "item_asset.csv",
            "item_subclass": "item_subclass.csv",
            "user_age": "user_age.csv",
            "user_region": "user_region.csv",
        }

        dataset = load_dataset(
            "KenBars/item_rec", data_files=data_files, data_dir="data"
        )

        for file_name in data_files:
            dataset[file_name].to_csv(f"data/{file_name}.csv")  # type: ignore
