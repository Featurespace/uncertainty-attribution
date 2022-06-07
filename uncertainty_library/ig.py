"""Functions to undertake IG computations"""

import numpy as np
import tensorflow as tf
import tqdm
from scipy.ndimage import gaussian_filter


# Pixel space vanilla IG
# =============================================================================

@tf.function
def d_entropy_d_x(x: tf.Tensor, model: tf.keras.Model):
    """Compute the derivative of Entropy wrt input features.

    Args:
        x (tf.Tensor): input image.
        model (tf.keras.Model): classifier.
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)

        # Score and entropy
        score = model(x[None, :])
        entr = - tf.reduce_sum(score * tf.math.log(score + 1e-30))

    # Diff wrt each pixel and return
    d_entr_d_x = tape.gradient(entr, x)
    return d_entr_d_x


def ig_entropy(
        fiducial: tf.Tensor,
        x: tf.Tensor,
        model: tf.keras.Model,
        bins=200):
    """Integrated gradients in pixel space.

    Args:
        fiducial (tf.Tensor): fiducial image
        x (tf.Tensor): real image
        model (tf.keras.Model): classifier
        bins (int, optional): Defaults to 200.
    """
    # Create the path from fiducial to x - half points - trapezoidal
    path = fiducial + \
        tf.range(1/bins/2, 1, 1/bins)[:, None, None, None] * (x - fiducial)

    # Cumulative trapezoidal integration, evaluate and sum each grid point
    integrand_entropy = tf.zeros(x.shape[1:], dtype=tf.float32)

    for i in tqdm.tqdm(range(bins)):
        entr = d_entropy_d_x(path[i], model)
        integrand_entropy += entr / bins

    # Weight by pixel value differences
    ig_entr = (x - fiducial) * integrand_entropy

    # Sum over channels
    ig_entr = tf.reduce_sum(ig_entr, axis=-1)
    return ig_entr.numpy()


# Latent level IG
# =============================================================================

@tf.function
def d_entropy_d_a_latent(
        a: tf.Tensor,
        z: tf.Tensor,
        z_fid: tf.Tensor,
        decoder: tf.keras.Model,
        model: tf.keras.Model):
    """Compute the derivative of Entropy wrt latent level features.

    Args:
        a (tf.Tensor): straight line.
        z (tf.Tensor): latent level real vector.
        z_fid (tf.Tensor): latent level fiducial.
        decoder (tf.keras.model): decoder model
        model (tf.keras.model): classifier
    """
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(a)

        # Procure state along latent path
        this_z = z_fid + a * (z - z_fid)

        # Decode latent vector into an image
        this_image = decoder(this_z)[0]

        # Score the image and compute the entropy
        score = model(this_image[None, ...])
        entr = - tf.reduce_sum(score * tf.math.log(score + 1e-30))

    # Derivative of entropy wrt decoded image
    d_entr_d_image = tape.gradient(entr, this_image)

    # Derivative of decoded image with respect to alpha point
    d_image_d_a = tf.reshape(
        tape.jacobian(this_image, a, parallel_iterations=256),
        shape=this_image.shape
    )

    # Compute values in pixel space
    aggr_pixel_space = (d_entr_d_image * d_image_d_a)
    return aggr_pixel_space


def ig_latent_curve(
        z_fid: tf.Tensor,
        z: tf.Tensor,
        x: tf.Tensor,
        decoder: tf.keras.Model,
        model: tf.keras.Model,
        bins=200):
    """Integrated gradients through a latent path

    Args:
        z_fid (tf.Tensor): ficudial in latent space
        z (tf.Tensor): real image in latent space
        x (tf.Tensor): real image
        decoder (tf.keras.Model): decoder model
        model (tf.keras.Model): classifier
        bins (int, optional): Defaults to 200.
    """

    # Discretise the [0, 1] range for the alpha parameter
    alpha = tf.range(1/bins/2, 1, 1/bins)[:, None]

    # Discretise straight path from decoded image to original
    path_interp = decoder(z) + \
        tf.range(1/bins/2, 1, 1/bins)[:, None, None, None] * (x - decoder(z))

    # Cumulative trapezoidal integration, evaluate and sum each grid point
    integrand_entropy = tf.zeros(x.shape[0:], dtype=tf.float32)
    integrand_entropy_interp = tf.zeros(x.shape[0:], dtype=tf.float32)

    # Loop and integrate numerically
    for i in tqdm.tqdm(range(bins)):
        entr = d_entropy_d_a_latent(alpha[i], z, z_fid, decoder, model)
        integrand_entropy += entr / bins

        entr = d_entropy_d_x(path_interp[i], model)
        integrand_entropy_interp += entr / bins

    # INTERPOLATION PART: Weight by pixel value differences, sum over channels
    ig_entr = (x - decoder(z)[0]) * integrand_entropy_interp

    # Add the latent path component which needs not weighting
    ig_entr += integrand_entropy    

    # Sum over channels
    ig_entr = tf.reduce_sum(ig_entr, axis=2)
    return ig_entr.numpy()


# Blur IG
# =============================================================================

def ig_entropy_blur(
        x: tf.Tensor,
        max_var: np.float,
        model: tf.keras.Model,
        bins=200):
    """IG in pixel space, through blurring path

    Args:
        x (tf.Tensor): real image
        max_var (np.float): maximum SD for blurring
        model (tf.keras.Model): classifier
        bins (int, optional): Defaults to 200.
    """
    # Create the path through reversed blurring
    range_var = tf.range(max_var-max_var/bins/2, 0, -max_var/bins)

    # Cumulative trapezoidal integration, evaluate and sum each grid point
    integrand_entropy = tf.zeros(x.shape[1:], dtype=tf.float32)

    for i in tqdm.tqdm(range(bins)):
        var = float(range_var[i])
        entr = d_entropy_d_x(gaussian_filter(x, sigma=(var, var, 0)), model)

        # Approximate derivative of filter wrt scale parameter
        filtered = gaussian_filter(x, sigma=(var, var, 0))
        filtered_h = gaussian_filter(x, sigma=(var+1e-10, var+1e-10, 0))
        d_filtered_d_var = (filtered - filtered_h)/1e-10
        integrand_entropy += (entr * d_filtered_d_var) * max_var / bins

    # Sum over channels
    ig_entr = tf.reduce_sum(integrand_entropy, axis=-1)

    return ig_entr.numpy()


# Guided IG
# =============================================================================

def ig_entropy_guided(
        fiducial: tf.Tensor,
        x: tf.Tensor,
        model: tf.keras.Model,
        p=0.1,
        steps=200):
    """Implementation of guided IG for Entropy attributions

    Args:
        fiducial (tf.Tensor): fiducial image
        x (tf.Tensor): real image
        model (tf.keras.Model): classifier
        p (float, optional): quantile level cut point. Defaults to 0.1.
        steps (int, optional): Defaults to 200.
    """
    d_tot = tf.abs((x-fiducial)).numpy().sum()  # Total distance to transverse
    state = tf.identity(fiducial).numpy()
    ig_entr = np.zeros_like(x)

    for t in range(steps):
        y = d_entropy_d_x(state, model).numpy()

        delta = np.inf
        while delta > 1:
            y[(state == x)] = np.inf
            d_target = d_tot * (1-(t+1)/steps)
            d_current = tf.abs(x-state).numpy().sum()
            if d_target == d_current:
                break

            cut = np.quantile(np.abs(y).flatten(), q=p)
            idx_S = np.abs(y) <= cut
            d_S = np.abs(x-state)[idx_S].sum()

            delta = (d_current - d_target) / d_S

            temp = state.copy()
            if delta > 1:
                state[idx_S] = x[idx_S]
            else:
                state[idx_S] = (1-delta) * state[idx_S] + delta * x[idx_S]

            y[y == np.inf] = 0
            ig_entr[idx_S] += y[idx_S] * (state[idx_S] - temp[idx_S])

    # Sum over channels
    ig_entr = ig_entr.sum(axis=-1)

    return ig_entr
