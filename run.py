"""Pipeline."""

import hydra
from omegaconf import DictConfig

from item_rec.models import IALS, RandomModel
from item_rec.preprocess import PreprocessData
from item_rec.metrics import Map10


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function which start the whole pipeline."""
    print("[PREPROCESING DATA] Start preprocessing Dataset...")
    preproc = PreprocessData(cfg["files"]["interactions"])
    dataset = preproc.get_data()

    print("[SPLIT DATA] Deviding into train and test Dataset...")
    train_dataset, test_usr_item = preproc.train_test_split(dataset)

    print("[TRAIN IALS] Start training IALS model...")
    model_ials = IALS(
        emb_size=cfg["ials_params"]["emb_size"],
        reg_coef=cfg["ials_params"]["reg_coef"],
        n_iter=cfg["ials_params"]["n_iter"],
    )
    users_emb, items_emb = model_ials.fit(train_dataset)

    print("[TRAIN RANDOM] Start training random model...")
    model_rand = RandomModel()
    model_rand.fit(train_dataset)

    print("[RECOMMEND IALS] Start recommend items using IALS model...")
    ials_recs = model_ials.predict()

    print("[RECOMMEND RANDOM] Start recommend items randomly...")
    rand_recs = model_rand.predict()

    print("[MAP_10 IALS] Start calculating of MAP_10 metric for IALS Model...")
    map_10_ials = Map10(ials_recs, test_usr_item)

    print("[MAP_10 Random] Start calculating of MAP_10 metric for Random Model...")
    map_10_rand = Map10(rand_recs, test_usr_item)

    print(f"MAP_10 trained in IALS : {map_10_ials.calculate_map_10()}")
    print(f"MAP_10 trained in Random : {map_10_rand.calculate_map_10()}")


if __name__ == "__main__":
    main()
