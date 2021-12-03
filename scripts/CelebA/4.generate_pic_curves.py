import os
import random

import numpy as np
import tensorflow as tf
from matplotlib import rcParams
from tqdm import tqdm
from uncertainty_library.attribution_methods import (get_importances,
                                                     get_importances_CLUE,
                                                     get_importances_LIME,
                                                     get_importances_SHAP,
                                                     get_importances_vanilla)
from uncertainty_library.data import read_format_data_CelebA
from uncertainty_library.plotting_functions import generate_pic_curves
from uncertainty_library.utils import ds2numpy, remove_test_time_dropout

rcParams["figure.facecolor"] = "white"


def generate_attributions(dir_name: str):
    for idx in tqdm(random.sample(range(len(x_test)), k=50)):
        x = x_test[idx]

        # Importances
        ig_entr, *_ = get_importances(x, model_drop, encoder, decoder, jacobian_splits=16)
        ig_entr_vanilla, *_ = get_importances_vanilla(x, model_drop)
        entr_clue = get_importances_CLUE(x, model_drop, encoder, decoder)
        entr_lime = get_importances_LIME(x, model_drop, bs=4, kernel_size=3,
                                         max_dist=10, ratio=2.0)
        entr_shap = get_importances_SHAP(x, model_drop, x_train, bs=4)

        attr_dict = {
            'image': x,
            'entr_ig_vae': ig_entr,
            'entr_ig_vanilla': ig_entr_vanilla,
            'entr_clue': entr_clue,
            'entr_lime': entr_lime,
            'entr_shap': entr_shap,
            'entr_random': np.random.uniform(size=x.shape[:2])
        }
        np.savez_compressed(os.path.join(dir_name, str(idx)), **attr_dict)


if __name__ == "__main__":
    # Load datasets
    ds_train, ds_test = read_format_data_CelebA('img_align_celeba', category='Smiling', bs=10)
    x_train, _ = ds2numpy(ds_train, max_num_batches=200)
    x_test, _ = ds2numpy(ds_test, max_num_batches=200)

    # Load the models
    encoder = tf.keras.models.load_model('models/dfc-vae-cnn/encoder')
    decoder = tf.keras.models.load_model('models/dfc-vae-cnn/decoder')
    model_drop = tf.keras.models.load_model('models/classifier_smiling')
    model = remove_test_time_dropout(model_drop)

    attr_bins_dir_name = "attr_binaries"
    pic_curves_dir_name = "pic_cruves"

    if not os.path.exists(attr_bins_dir_name):
        os.makedirs(attr_bins_dir_name)
        generate_attributions(attr_bins_dir_name)

    os.makedirs(pic_curves_dir_name, exist_ok=True)
    generate_pic_curves(attr_bins_dir_name, pic_curves_dir_name, model, "entropy", blur=4)
