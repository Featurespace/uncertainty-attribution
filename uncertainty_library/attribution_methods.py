"""Methods to assign uncertainty attributions to input pixels in images"""

import copy

import numpy as np
import skimage
import skimage.segmentation
import sklearn.metrics
import tensorflow as tf
from sklearn import linear_model

from .ig import (
    ig_entropy,
    ig_latent_curve,
    ig_entropy_blur,
    ig_entropy_guided
)
from .latent_optimisation import (
    get_clue_counterfactual,
    get_in_class_fiducial,
    get_latent_representation
)


# Integrated gradients variants
# =============================================================================

def get_importances_VAE(
        x: tf.Tensor,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        known_class=None,bins=200):
    """IG attributions with integration path in latent space

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        encoder (tf.keras.Model): image encoder
        decoder (tf.keras.Model): image decoder
        known_class (int, optional): class tag. Defaults to None.
        bins (int, optional): interpolation bins. Defaults to 200.
    """
    z = get_latent_representation(x, encoder, decoder)

    # Obtain a fiducial and measure the entropy reduction achieved
    z_fid = None
    x_ent = get_entropy(x[None, ...], model)
    z_fid = get_in_class_fiducial(
        x,
        model,
        encoder,
        decoder,
        constrain_multiplier=1e4,
        known_class=known_class)
    fid_ent = get_entropy(decoder(z_fid), model)

    # Create a dictionary with relevant info and return
    log = {}
    log['latent_representation'] = z
    log['latent_fiducial'] = z_fid
    log['uncertainty reduction'] = x_ent - fid_ent
    log['perc. uncertainty reduction'] = (1 - fid_ent / x_ent)
    return ig_latent_curve(z_fid, z, x, decoder, model, bins=bins), log


def get_importances_vanilla(x: tf.Tensor, model, bins=200):
    """Produce vanilla IG attributions of uncertainties.

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        bins (int, optional): interpolation bins. Defaults to 200.
    """
    fiducial = tf.zeros_like(x)
    return ig_entropy(fiducial, x, model, bins=bins)


def get_importances_vanilla_bw(x: tf.Tensor, model, bins=200):
    """Produce B+W IG attributions of uncertainties.

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        bins (int, optional): interpolation bins. Defaults to 200.
    """
    fiducial_black = tf.zeros_like(x)
    fiducial_white = tf.ones_like(x)

    # Average importances and return
    ig_black = ig_entropy(fiducial_black, x, model, bins)
    ig_white = ig_entropy(fiducial_white, x, model, bins)
    return (ig_black + ig_white)/2


def get_importances_vanilla_counter(
        x: tf.Tensor,
        counterfactual: tf.Tensor,
        model: tf.keras.Model,
        bins=200):
    """Produce IG attributions of uncertainties, with counterfactual fiducial.

    Args:
        x (tf.Tensor): evaluation data point or image.
        counterfactual (tf.Tensor): counterfactual image.
        model (tf.keras.Model): tensorflow calssification model
        bins (int, optional): interpolation bins. Defaults to 200.
    """
    return ig_entropy(counterfactual, x, model, bins=bins)


def get_importances_blur(
        x: tf.Tensor,
        max_var: np.float,
        model: tf.keras.Model,
        bins=200):
    """Produce Blur IG attributions of uncertainties.

    Args:
        x (tf.Tensor): evaluation data point or image.
        max_var (np.float): blurring maximum standard deviation setting.
        model (tf.keras.Model): tensorflow calssification model
        bins (int, optional): interpolation bins. Defaults to 200.
    """
    return ig_entropy_blur(x, max_var, model, bins=bins)


def get_importances_guided(x: tf.Tensor, model: tf.keras.Model, steps=200):
    """Guided IG importances with a black image fiducial.

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        steps (int, optional): guided IG steps. Defaults to 200.
    """
    fiducial = tf.zeros_like(x)
    return ig_entropy_guided(fiducial, x, model, steps=steps)


def get_importances_guided_bw(x: tf.Tensor, model: tf.keras.Model, steps=200):
    """Guided IG importances with B+W fiducial setting.

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        steps (int, optional): guided IG steps. Defaults to 200.
    """
    fiducial_black = tf.zeros_like(x)
    fiducial_white = tf.ones_like(x)

    # Get importances, average and return
    ig_black = ig_entropy_guided(fiducial_black, x, model, steps=steps)
    ig_white = ig_entropy_guided(fiducial_white, x, model, steps=steps)
    return (ig_black + ig_white)/2


def get_importances_guided_counter(
        x: tf.Tensor,
        counterfactual: tf.Tensor,
        model: tf.keras.Model,
        steps=200):
    """Guided IG importances with counterfactual fiducial setting.

    Args:
        x (tf.Tensor): evaluation data point or image.
        counterfactual (tf.Tensor): counterfactual image.
        model (tf.keras.Model): tensorflow calssification model
        steps (int, optional): guided IG steps. Defaults to 200.
    """
    return ig_entropy_guided(counterfactual, x, model, steps=steps)


# Pure counterfactual methods
# =============================================================================

def get_importances_CLUE(
        x: tf.Tensor,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model):
    """CLUE attributions.

    Args:
        x (tf.Tensor): evaluation data point or image.
        model (tf.keras.Model): tensorflow calssification model
        encoder (tf.keras.Model): image encoder
        decoder (tf.keras.Model): image decoder
    """
    z_fid = None
    z_fid = get_clue_counterfactual(x, model, encoder, decoder)
    return tf.reduce_sum(x - decoder(z_fid)[0], axis=2).numpy()


# Segmentation and re-sampling methods
# =============================================================================

def get_importances_LIME(
        x: tf.Tensor,
        model: tf.keras.Model,
        num_perturb=None,
        kernel_size=1,
        max_dist=5,
        ratio=0.2):
    """Implementation of LIME importances with quickshift segmentation.

    Args:
        x (tf.Tensor): input image.
        model (tf.keras.Model): classification model
        num_perturb (_type_, optional): number perturbations. Defaults to None.
        kernel_size (int, optional): segmentation kernel size. Defaults to 1.
        max_dist (int, optional): max distance for segmentation. Defaults to 5.
        ratio (float, optional): ratio for segmentation. Defaults to 0.2.
    """

    # Generate superpixels
    channels = x.shape[2]
    if channels == 1:
        x_rgb = skimage.color.gray2rgb(x[:, :, 0]).astype('double')
    else:
        x_rgb = x.astype('double')
    superpixels = skimage.segmentation.quickshift(
        x_rgb, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
    num_superpixels = np.unique(superpixels).shape[0]

    if num_perturb is None:
        num_perturb = int((10 * num_superpixels + 100)/2)

    # Generate perturbations
    perturbations = np.random.binomial(
        1, 0.8, size=(num_perturb, num_superpixels))

    # Generate predictions for perturbations
    perturbed_images = []
    for pert in perturbations:
        perturbed_images.append(
            perturb_image(x_rgb, pert, superpixels)[None, :, :, :channels]
        )
    perturbed_images = tf.concat(values=perturbed_images, axis=0)
    predictions = get_entropy(perturbed_images, model)

    # Regression weights
    reference = np.ones(num_superpixels)[None, :]
    w = sklearn.metrics.pairwise_distances(perturbations,
                                           reference,
                                           metric='cosine').ravel()
    w = np.exp(-w)

    # Run Lasso regression
    lf = linear_model.LassoCV(cv=5)
    lf.fit(X=perturbations, y=predictions, sample_weight=w)

    # Get importances in original image shape
    return lf.coef_[superpixels]


def get_importances_SHAP(
    x: tf.Tensor,
    model: tf.keras.Model,
    x_train: tf.Tensor,
    num_perturb=None):
    """Implementation of the kernelShap attribution method.

    Args:
        x (tf.Tensor): input image.
        model (tf.keras.Model): classification model.
        x_train (tf.Tensor): train data.
        num_perturb (_type_, optional): perturbations. Defaults to None.
    """
    if num_perturb is None:
        num_perturb = 3 * x.shape[0] ** 2 + 2 ** 11

    # Generate perturbations
    perc = 0.05
    feat_c = x.shape[0]*x.shape[1]
    perturbations = np.repeat(x[None, :], num_perturb, axis=0)
    mask = np.ones(shape=perturbations.shape[:3])

    # Sample from marginal x_train
    deac_0 = np.repeat(range(num_perturb), int(feat_c*perc))
    deac_1 = np.random.choice(x.shape[0], deac_0.shape[0], replace=True)
    deac_2 = np.random.choice(x.shape[1], deac_0.shape[0], replace=True)
    idx_train = np.random.choice(x_train.shape[0],
                                 deac_0.shape[0], replace=True)

    perturbations[deac_0, deac_1, deac_2] = \
        x_train[idx_train, deac_1, deac_2].copy()
    mask[deac_0, deac_1, deac_2] = 0

    # Get entropies of perturbations
    entropies = get_entropy(perturbations, model)

    # Run regression
    lf = linear_model.LassoLarsIC(criterion='aic', normalize=True)
    lf.fit(X=mask.reshape(num_perturb, -1), y=entropies)

    # Get importances in original image shape
    return lf.coef_.reshape(x.shape[:2])


def get_importances_SHAP_counter(
        x: tf.Tensor,
        model: tf.keras.Model,
        counterfactual: tf.Tensor,
        num_perturb=None):
    """Implementation of the kernelShap attribution method.
        1. Resampling replaces real pixels with counterfactual image pixels.

    Args:
        x (tf.Tensor): input image.
        model (tf.keras.Model): classification model.
        counterfactual (tf.Tensor): counterfactual image.
        num_perturb (_type_, optional): perturbations. Defaults to None.
    """
    if num_perturb is None:
        num_perturb = 3 * x.shape[0] ** 2  + 2 ** 11

    # Generate perturbations
    perc = 0.05
    feat_c = x.shape[0]*x.shape[1]
    perturbations = np.repeat(x[None, :], num_perturb, axis=0)
    mask = np.ones(shape=perturbations.shape[:3])

    # Replace by pixels in the counterfactual
    deac_0 = np.repeat(range(num_perturb), int(feat_c*perc))
    deac_1 = np.random.choice(x.shape[0], deac_0.shape[0], replace=True)
    deac_2 = np.random.choice(x.shape[1], deac_0.shape[0], replace=True)

    perturbations[deac_0, deac_1, deac_2] = \
        counterfactual.numpy()[deac_1, deac_2].copy()
    mask[deac_0, deac_1, deac_2] = 0

    # Get entropies of perturbations
    entropies = get_entropy(perturbations, model)

    # Run regression
    lf = linear_model.LassoLarsIC(criterion='aic', normalize=True)
    lf.fit(X=mask.reshape(num_perturb, -1), y=entropies)

    # Get importances in original image shape
    return lf.coef_.reshape(x.shape[:2])


# Support functions
# =============================================================================

def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1

    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, None]

    return perturbed_image


def get_entropy(x, model):
    # x assumed to be a batch of images
    ds = tf.data.Dataset.from_tensor_slices(x).batch(4)
    preds = model.predict(ds)
    entropies = -np.sum(preds * np.log(preds + 1e-30), axis=-1)
    return entropies
