import numpy as np
import tensorflow as tf
import skimage
import skimage.segmentation
import copy
import sklearn.metrics
from sklearn import linear_model

from .mc_drop import get_entropy
from .latent_optimisation import (
    get_latent_representation,
    get_in_class_fiducial,
    get_clue_counterfactual,
)
from .ig import (
    ig_entropy,
    ig_latent_curve,
)


# OUR PROPOSED METHOD
def get_importances(x: tf.Tensor, model, encoder, decoder, known_class=None,
                    max_num_trials=3, entropy_reduction=0.90, jacobian_splits=1):
    """IG attributions with integration path in latent space.
    If getting OOM error, increase the jacobian_splits parameter.
    Make sure that the image size is divisible by jacobian_splits."""

    if x.shape[0] % jacobian_splits:
        raise ValueError(f"image size ({x.shape[0]}) is not divisible by "
                         f"`jacobian_splits` ({jacobian_splits}).")

    z = get_latent_representation(x, encoder, decoder)

    # If the fiducial doesn't reach low enough entropy,
    # run the optimisation algorithm again. Do it at most
    # `max_num_trials` times.
    z_fid = None
    x_ent = get_entropy(x[None, ...], model)
    for _ in range(max_num_trials):
        z_fid = get_in_class_fiducial(x, model, encoder, decoder, known_class=known_class)
        fid_ent = get_entropy(decoder(z_fid), model)
        if (fid_ent <= (1 - entropy_reduction) * x_ent):
            break

    return ig_latent_curve(z_fid, z, x, decoder, model, jacobian_splits=jacobian_splits)


def get_importances_vanilla(x: tf.Tensor, model):
    # Set fiducial to a black image
    fiducial = tf.zeros_like(x)

    return ig_entropy(fiducial, x, model)


def get_importances_CLUE(x: tf.Tensor, model, encoder, decoder,
                         max_num_trials=5, entropy_reduction=0.90):
    # If the fiducial doesn't reach low enough entropy,
    # run the optimisation algorithm again. Do it at most
    # `max_num_trials` times.
    z_fid = None
    x_ent = get_entropy(x[None, ...], model)
    for _ in range(max_num_trials):
        z_fid = get_clue_counterfactual(x, model, encoder, decoder)
        fid_ent = get_entropy(decoder(z_fid), model)
        if (fid_ent <= (1 - entropy_reduction) * x_ent):
            break

    return tf.reduce_sum(x - decoder(z_fid)[0], axis=2).numpy()


def get_importances_LIME(x: tf.Tensor, model, mc_sample=100, num_perturb=100,
                         alpha=0.001, bs=32):
    # Generate superpixels
    channels = x.shape[2]
    if channels == 1:
        x_rgb = skimage.color.gray2rgb(x[:, :, 0]).astype('double')
    else:
        x_rgb = x.astype('double')
    superpixels = skimage.segmentation.quickshift(x_rgb, kernel_size=1,
                                                  max_dist=5, ratio=0.2)
    num_superpixels = np.unique(superpixels).shape[0]

    # Generate perturbations
    perturbations = np.random.binomial(1, 0.8, size=(num_perturb,
                                                     num_superpixels))

    # Generate predictions for perturbations
    perturbed_images = []
    for pert in perturbations:
        perturbed_images.append(
            perturb_image(x_rgb, pert, superpixels)[None, :, :, :channels]
        )
    perturbed_images = tf.concat(perturbed_images, axis=0)
    predictions = get_entropy(perturbed_images, model, mc_sample, bs=bs)

    # Regression weights
    reference = np.ones(num_superpixels)[None, :]
    w = sklearn.metrics.pairwise_distances(perturbations,
                                           reference,
                                           metric='cosine').ravel()
    w = np.exp(-w)

    # Run Lasso regression
    lf = linear_model.Lasso(alpha=alpha)
    lf.fit(X=perturbations, y=predictions, sample_weight=w)

    # Get importances in original image shape
    return lf.coef_[superpixels]


def get_importances_SHAP(x: tf.Tensor, model, x_train, mc_sample=100,
                         num_perturb=5000, alpha=0.002, bs=32):
    """Implementation of the kernelShap attribution method."""
    # Generate perturbations
    perc = 0.05
    feat_c = x.shape[0]*x.shape[1]
    perturbations = np.repeat(x[None, :], num_perturb, axis=0)
    mask = np.ones(shape=perturbations.shape[:3])

    # Sample from marginal x_train
    deac_0 = np.repeat(range(num_perturb), int(feat_c*perc))
    deac_1 = np.random.choice(x.shape[0], deac_0.shape[0], replace=True)
    deac_2 = np.random.choice(x.shape[0], deac_0.shape[0], replace=True)
    idx_train = np.random.choice(x_train.shape[0],
                                 deac_0.shape[0], replace=True)

    perturbations[deac_0, deac_1, deac_2] = \
        x_train[idx_train, deac_1, deac_2].copy()
    mask[deac_0, deac_1, deac_2] = 0

    # Score the perturbations
    predictions = get_entropy(perturbations, model, mc_sample, bs=bs)

    # Run regression
    lf = linear_model.Lasso(alpha=alpha)
    lf.fit(X=mask.reshape(num_perturb, -1),
           y=predictions)

    # Get importances in original image shape
    return lf.coef_.reshape(x.shape[:2])


def perturb_image(img, perturbation, segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1

    perturbed_image = copy.deepcopy(img)
    perturbed_image = perturbed_image * mask[:, :, None]

    return perturbed_image
