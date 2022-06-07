import os

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import rcParams
from tqdm import tqdm
from sklearn import metrics
from uncertainty_library.attribution_methods import (
    get_importances_VAE,
    get_importances_CLUE,
    get_importances_LIME,
    get_importances_SHAP,
    get_importances_SHAP_counter,
    get_importances_vanilla,
    get_importances_vanilla_bw,
    get_importances_vanilla_counter,
    get_importances_blur,
    get_importances_guided,
    get_importances_guided_bw,
    get_importances_guided_counter)
from uncertainty_library.xrai import get_segments_felzenszwalb, xrai_full
from uncertainty_library.data import read_format_data_CelebA
from uncertainty_library.utils import ds2numpy
from uncertainty_library.metrics import from_image_remove_pixels

rcParams["figure.facecolor"] = "white"
METHODS = [
    'vae',
    'ig', 'ig_bw', 'ig_counter', 
    'blur_ig', 'guided_ig', 'guided_ig_bw', 'guided_ig_counter',
    'lime',
    'shap', 'shap_counter',
    'clue',
    'xrai_vae', 'xrai_ig', 'xrai_ig_bw', 'xrai_ig_counter',
    'xrai_gig', 'xrai_gig_bw', 'xrai_gig_counter',
    'random']


def generate_attributions(
        x_train: np.ndarray,
        x_test: np.ndarray,
        dir_name: str,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model):
    importances = {}
    for method in METHODS:
        importances[method] = np.zeros(
            shape=[x_test.shape[0], *x_test.shape[1:3]])

    for idx, x in enumerate(tqdm(x_test)):
        # Importances   
        print(idx)

        # Generative
        importances['vae'][idx], log = get_importances_VAE(
            x, model, encoder, decoder, bins=25)
        counterfactual = decoder(log['latent_fiducial'])[0]

        # IG
        importances['ig'][idx] = get_importances_vanilla(x, model, bins=25)
        importances['ig_bw'][idx] = get_importances_vanilla_bw(
            x, model, bins=25)
        importances['ig_counter'][idx] = get_importances_vanilla_counter(
            x, counterfactual, model, bins=25)
        
        # IG main variants
        importances['blur_ig'][idx] = get_importances_blur(
            x, max_var=2, model=model, bins=25)
        importances['guided_ig'][idx] = get_importances_guided(
            x, model, steps=25)
        importances['guided_ig_bw'][idx] = get_importances_guided_bw(
            x, model, steps=25)
        importances['guided_ig_counter'][idx] = get_importances_guided_counter(
            x, counterfactual, model, steps=25)

        # Lime
        importances['lime'][idx] = get_importances_LIME(
            x, model, kernel_size=3, max_dist=10, ratio=2.0)

        # CLUE
        importances['clue'][idx] = get_importances_CLUE(
            x, model, encoder, decoder)
        
        # Random
        importances['random'][idx] = np.random.uniform(size=x.shape[:2])

        # Xrai
        print("segmenting...")
        segments = get_segments_felzenszwalb(x)
        importances['xrai_vae'][idx] = xrai_full(
            importances['vae'][idx], segments)
        importances['xrai_ig'][idx] = xrai_full(
            importances['ig'][idx], segments)
        importances['xrai_ig_bw'][idx] = xrai_full(
            importances['ig_bw'][idx], segments)
        importances['xrai_ig_counter'][idx] = xrai_full(
            importances['ig_counter'][idx], segments)
        importances['xrai_gig'][idx] = xrai_full(
            importances['guided_ig'][idx], segments)
        importances['xrai_gig_bw'][idx] = xrai_full(
            importances['guided_ig_bw'][idx], segments)
        importances['xrai_gig_counter'][idx] = xrai_full(
            importances['guided_ig_counter'][idx], segments)

    np.savez_compressed(
        os.path.join(dir_name, "importances_and_images"),
        **importances,
        images=x_test
    )


def compute_performance(binaries_dir_name: str, model: tf.keras.Model, labels: np.array,
                        pixel_removal_type: str, blur_sd: float = 1):
    # Load importances and images
    importances = dict(np.load(os.path.join(binaries_dir_name, "importances_and_images.npz")))
    images = importances.pop('images')

    # compute performances
    eic = {}
    aic = {}
    urc = {}
    ent = {}
    sp_ent = {}
    fl_sz = {}
    for method in METHODS:
        print(method)
        eic[method], aic[method], urc[method], ent[method], sp_ent[method], fl_sz[method] = \
            from_image_remove_pixels(
                images,
                importances[method],
                model,
                labels,
                pixel_removal_type,
                blur_sd,
                entropy=True,
            )

    # Save performances
    np.savez_compressed(os.path.join(binaries_dir_name, "eic"), **eic)
    np.savez_compressed(os.path.join(binaries_dir_name, "aic"), **aic)
    np.savez_compressed(os.path.join(binaries_dir_name, "urc"), **urc)
    np.savez_compressed(os.path.join(binaries_dir_name, "ent"), **ent)
    np.savez_compressed(os.path.join(binaries_dir_name, "sp_ent"), **sp_ent)
    np.savez_compressed(os.path.join(binaries_dir_name, "fl_sz"), **fl_sz)


def summarize_performance(binaries_dir_name: str):
    # Load precomputed performances
    eic = dict(np.load(os.path.join(binaries_dir_name, "eic.npz")))
    urc = dict(np.load(os.path.join(binaries_dir_name, "urc.npz")))
    ent = dict(np.load(os.path.join(binaries_dir_name, "ent.npz")))
    sp_ent = dict(np.load(os.path.join(binaries_dir_name, "sp_ent.npz")))

    # Compute area under the curve for information curves
    auc= pd.DataFrame(
        columns=[
            'EIC_pixel',
            'EIC_shannon',
            'EIC_spatial'
        ], index=list(eic.keys()))
    for method in METHODS:
        auc.loc[method, 'EIC_pixel'] = \
            1-metrics.auc(np.arange(0, 1, 1/eic[method].shape[0]), eic[method])

        x_ax = ent[method] - 1
        x_ax = np.maximum.accumulate(x_ax[::-1])[::-1]
        x_ax += np.linspace(1e-10, 0, x_ax.shape[0])
        x_ax = x_ax / x_ax[0]
        auc.loc[method, 'EIC_shannon'] = 1-metrics.auc(x_ax, eic[method])

        x_ax = sp_ent[method] - 1
        x_ax = np.maximum.accumulate(x_ax[::-1])[::-1]
        x_ax += np.linspace(1e-10, 0, x_ax.shape[0])
        x_ax = x_ax / x_ax[0]
        auc.loc[method, 'EIC_spatial'] = 1-metrics.auc(x_ax, eic[method])

    auc.to_csv(f'performance_results/auc.csv')

    # Percentile points while blurring pixels from original image
    perc= pd.DataFrame(
        columns=[
            'EIC_pixel_5%', 'EIC_pixel_10%',
            'EIC_shannon_5%', 'EIC_shannon_10%',
            'EIC_spatial_5%', 'EIC_spatial_10%',],
        index=list(eic.keys()))

    for method in METHODS:
        perc.loc[method, 'EIC_pixel_5%'] = \
            (eic[method] / eic[method][-1])[::-1][
                int(np.round(eic[method].shape[0] * 0.05))]
        perc.loc[method, 'EIC_pixel_10%'] = \
            (eic[method] / eic[method][-1])[::-1][
                int(np.round(eic[method].shape[0] * 0.1))]

        x_ax = ent[method] - 1
        x_ax = x_ax / x_ax[0]
        loss_target = np.where(x_ax < 0.05)[0][0]
        perc.loc[method, 'EIC_shannon_5%'] = \
            (eic[method] / eic[method][-1])[loss_target]
        loss_target = np.where(x_ax < 0.10)[0][0]
        perc.loc[method, 'EIC_shannon_10%'] = \
            (eic[method] / eic[method][-1])[loss_target]

        x_ax = sp_ent[method] - 1
        x_ax = x_ax / x_ax[0]
        loss_target = np.where(x_ax < 0.05)[0][0]
        perc.loc[method, 'EIC_spatial_5%'] = \
            (eic[method] / eic[method][-1])[loss_target]
        loss_target = np.where(x_ax < 0.10)[0][0]
        perc.loc[method, 'EIC_spatial_10%'] = \
            (eic[method] / eic[method][-1])[loss_target]

    perc.to_csv(f'performance_results/eic_at_percentiles.csv')


    # Compute perc uncertainty reduced at cut points
    urc_perc= pd.DataFrame(
        columns=[
            'pixel_5%', 'pixel_10%',
            'shannon_5%', 'shannon_10%',
            'spatial_5%', 'spatial_10%',], index=list(eic.keys()))
    for method in METHODS:
        urc_perc.loc[method, 'pixel_5%'] = \
            urc[method][np.round(0.05*urc[method].shape[0]).astype(int)]
        urc_perc.loc[method, 'pixel_10%'] = \
            urc[method][np.round(0.10*urc[method].shape[0]).astype(int)]

        x_ax = ent[method] - 1
        x_ax = x_ax / x_ax[0]
        loss_target = np.where(x_ax < 0.05)[0][0]
        urc_perc.loc[method, 'shannon_5%'] = \
            urc[method][urc[method].shape[0] - 1 - loss_target]
        loss_target = np.where(x_ax < 0.10)[0][0]
        urc_perc.loc[method, 'shannon_10%'] = \
            urc[method][urc[method].shape[0] - 1 - loss_target]

        x_ax = sp_ent[method] - 1
        x_ax = x_ax / x_ax[0]
        loss_target = np.where(x_ax < 0.05)[0][0]
        urc_perc.loc[method, 'spatial_5%'] = \
            urc[method][urc[method].shape[0] - 1 - loss_target]
        loss_target = np.where(x_ax < 0.10)[0][0]
        urc_perc.loc[method, 'spatial_10%'] = \
            urc[method][urc[method].shape[0] - 1 - loss_target]

    urc_perc.to_csv(f'performance_results/urc_at_percentiles.csv')


if __name__ == "__main__":
    # Load datasets
    train_ds, valid_ds = read_format_data_CelebA(
        'img_align_celeba',
        category='Smiling', bs=10)
    x_train, y_train = ds2numpy(train_ds, max_num_batches=10)
    x_test, y_test = ds2numpy(valid_ds, max_num_batches=40)

    # Load the models
    encoder = tf.keras.models.load_model(
        'models/dfc-vae-cnn/encoder')
    decoder = tf.keras.models.load_model(
        'models/dfc-vae-cnn/decoder')
    model = tf.keras.models.load_model(
        'models/classifier_smiling')

    attr_bins_dir_name = "attr_binaries/celeba_smiling" 

    # Generate and save attribution maps
    os.makedirs(attr_bins_dir_name, exist_ok=True)

    generate_attributions(x_train, x_test, attr_bins_dir_name,
                            model, encoder, decoder)

    # Compute and save performance scores
    compute_performance(
        binaries_dir_name=attr_bins_dir_name,
        model=model,
        labels = y_test,
        pixel_removal_type = 'blur',
        blur_sd = 3)

    # # Produce and save key metrics
    os.makedirs("performance_results", exist_ok=True)
    summarize_performance(attr_bins_dir_name)
