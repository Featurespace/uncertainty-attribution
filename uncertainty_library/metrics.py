""" Methods to compute EIC, AIC and URC performance metrics """

from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from scipy.ndimage import gaussian_filter
from skimage.measure import shannon_entropy
from uncertainty_library.webp_entropy import get_size_pil


# EIC, AIC, URC
# =============================================================================

def from_image_remove_pixels(
        images: np.array,
        attr: np.array,
        model: tf.keras.Model,
        labels: np.array,
        pixel_removal_type='blur',
        blur_sd = 1.0,
        entropy=True):
    """Start with the original image:
        1. Gradually blur/gray pixels which most CONTRIBUTE TO uncertainty
        2. Evaluate change on uncertainty and accuracy of the predicted vector
        3. Compute performance metrics

    Args:
        images (np.array): collection of images to evaluate
        attr (np.array): uncertainty importances for the images
        model (tf.keras.Model): classifier
        labels (np.array): observed labels
        pixel_removal_type (str, optional): Blur or gray. Defaults to 'blur'.
        blur_sd (float, optional): Defaults to 1.0.
        entropy (bool, optional): Compute entropy measures? Defaults to True.
    """
    if pixel_removal_type == 'blur':
        ref = gaussian_filter(images, sigma=(0, blur_sd, blur_sd, 0))
    elif pixel_removal_type == 'gray':
        ref = np.full(images.shape, fill_value=0.5)
    else:
        raise Exception("type must be 'blurred' or 'gray'")

    # Retrieve observed label
    y = labels.argmax(axis=1)

    # Break degeneracy, i.e. same value across pixels can't be ordered
    attr += np.random.normal(loc=0, scale=1e-100, size=attr.shape)

    # Rank pixels by contribution to uncertainty, per image
    attr_temp = (-attr).reshape(images.shape[0], -1)
    order_pixels = np.argsort(attr_temp, axis=1)

    # Gradually remove pixels, 1 at a time
    eic, aic, ent, sp_ent, fl_sz = \
        _remove_pixels(images, y, ref, order_pixels, model, entropy)

    if entropy:
        ent = ent / ent[:, 0][:, None]
        sp_ent = sp_ent / sp_ent[:, 0][:, None]
        fl_sz = fl_sz / fl_sz[:, 0][:, None]

    # Aggregate relative measures
    eic_from_x = eic / eic[:, 0][:, None]
    eic_from_x = 1 - eic_from_x
    eic = eic / eic[:, -1][:, None]

    # Get the mean of that
    eic_av = np.median(eic, axis=0)[::-1]  # Adding pixels from blurred 
    aic_av = np.mean(aic, axis=0)[::-1]

    ent_av = np.median(ent, axis=0)[::-1]  # Adding pixels from blurred 
    sp_ent_av = np.median(sp_ent, axis=0)[::-1]
    fl_sz_av = np.median(fl_sz, axis=0)[::-1]

    # X% uncertainty reduction attained by percentage of pixels removed
    urc_av = _cummax_per_entropy_reduced(eic_from_x)

    return eic_av, aic_av, urc_av, ent_av, sp_ent_av, fl_sz_av


def _remove_pixels(
        x: np.array,
        y: np.array,
        ref: np.array,
        order_pixels: np.array,
        model: tf.keras.Model,
        entropy=True,
        num_workers=16):
    """Remove pixels, one by one:
        1. Store change in uncertainty as a consequence

    Args:
        x (np.array): input images
        y (np.array): labels
        ref (np.array): reference image after blurring or gray
        order_pixels (np.array): order of pixel removal
        model (tf.keras.Model): classifier
        entropy (bool, optional): Compute entropy metrics? Defaults to True.
        num_workers (int, optional): Defaults to 16.
    """
    dummy_x = x.copy()
    eic = np.zeros(shape=(order_pixels.shape[0], order_pixels.shape[1]+1))
    aic = np.zeros(shape=(order_pixels.shape[0], order_pixels.shape[1]+1))
    ent = np.zeros(shape=(order_pixels.shape[0], order_pixels.shape[1]+1))
    sp_ent = np.zeros(shape=(order_pixels.shape[0], order_pixels.shape[1]+1))
    fl_sz = np.zeros(shape=(order_pixels.shape[0], order_pixels.shape[1]+1))

    # Real image output vector
    pred_dummy = model(dummy_x)

    # Store real image predictive entropy, score and information
    eic[:, 0] = -np.sum(pred_dummy * np.log(pred_dummy + 1e-30), axis=1)
    aic[:, 0] = (np.argmax(pred_dummy, axis=1) == y).astype(float)

    if entropy:
        ent[:, 0] = [
            shannon_entropy(
                dummy_x[j], base=2
            ) for j in range(dummy_x.shape[0])]
        grad_x = np.gradient(dummy_x, axis=1)
        sp_ent[:, 0] = [
            shannon_entropy(
                grad_x[j], base=2
            ) for j in range(dummy_x.shape[0])]
        fl_sz[:, 0] = [
            get_size_pil(
                dummy_x[j], format='png'
            ) for j in range(dummy_x.shape[0])]

    # Loop over pixels
    with Pool(num_workers) as p:
        for i in tqdm.tqdm(range(order_pixels.shape[1])):
            pixels = order_pixels[:, i]
            pixel_idx = np.array(np.unravel_index(pixels, x.shape[1:3]))

            # Change pixel for reference
            dummy_x[np.arange(x.shape[0]), pixel_idx[0], pixel_idx[1]] = \
                ref[np.arange(x.shape[0]), pixel_idx[0], pixel_idx[1]]

            # Predict and store values
            pred_dummy = model(dummy_x)
            eic[:, i+1] = -np.sum(
                pred_dummy * np.log(pred_dummy + 1e-30), axis=1)
            aic[:, i+1] = (np.argmax(pred_dummy, axis=1) == y).astype(float)

            if entropy:
                ent[:, i+1] = p.map(partial(shannon_entropy, base=2), dummy_x)
                grad_x = np.gradient(dummy_x, axis=1)
                sp_ent[:, i+1] = p.map(
                    partial(shannon_entropy, base=2), grad_x)
                fl_sz[:, i+1] = [
                    get_size_pil(
                        dummy_x[j],
                        format='png'
                    ) for j in range(dummy_x.shape[0])]

    return eic, aic, ent, sp_ent, fl_sz


def _cummax_per_entropy_reduced(eic_from_x: np.array):
    """How much do we reduce undertainty?
        1. As we gradually remove pixels

    Args:
        eic_from_x (np.array): eic metric starting at original image
    """

    out = pd.DataFrame(
        eic_from_x.T,
        index = np.arange(0, eic_from_x.shape[1])
    )
    out.index.name = 'Pixels_Removed'
    out = out.cummax()
    out = out.median(axis=1)
    out.name = 'Reduction_Uncertainty'

    return out
