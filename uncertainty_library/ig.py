import tensorflow as tf
import tqdm

from .latent_optimisation import (
    d_entropy_d_z,
    get_jacobian
)


@tf.function
def d_entropy_d_x(x, model, sample):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)

        # MC samples of scores
        scores = model(tf.repeat(x[None, :], sample, axis=0))

        # Compute the entropy of expected score
        score_m = tf.reduce_mean(scores, axis=0)
        entr = - tf.reduce_sum(score_m * tf.math.log(score_m + 1e-30))

        # Compute expected entropy across scores
        entropies = tf.reduce_sum(scores * tf.math.log(scores + 1e-30), axis=1)
        entr_ale = - tf.reduce_mean(entropies)

    # Diff wrt each pixel and return
    d_entr_d_x = tape.gradient(entr, x)
    d_ale_d_x = tape.gradient(entr_ale, x)
    return d_entr_d_x, d_ale_d_x


def ig_entropy(fiducial, x, model, bins=500, sample=500):
    """IG in pixel space"""

    # Create the path from fiducial to x - half points - trapezoidal
    path = fiducial + tf.range(1/bins/2, 1, 1/bins)[:, None, None, None] * (x - fiducial)

    # Cumulative trapezoidal integration, evaluate and sum each grid point
    integrand_entropy = tf.zeros(x.shape[1:], dtype=tf.float32)
    integrand_aleatoric = tf.zeros(x.shape[1:], dtype=tf.float32)

    for i in tqdm.tqdm(range(bins)):
        entr, ale = d_entropy_d_x(path[i], model, sample)
        integrand_entropy += entr / bins
        integrand_aleatoric += ale / bins

    # Weight by pixel value differences
    ig_entr = (x - fiducial) * integrand_entropy
    ig_ale = (x - fiducial) * integrand_aleatoric

    # Sum over channels
    ig_entr = tf.reduce_sum(ig_entr, axis=-1)
    ig_ale = tf.reduce_sum(ig_ale, axis=-1)
    ig_epi = ig_entr - ig_ale

    return ig_entr.numpy(), ig_ale.numpy(), ig_epi.numpy()


# =============================================================================

def ig_latent_curve(z_fid, z, x, decoder, model, bins=500, sample=500, jacobian_splits=1):
    """IG with counterfactual fiducial and latent path"""

    # Create the paths, for trapezoidal integration
    path = z_fid + tf.range(1/bins/2, 1, 1/bins)[:, None] * (z - z_fid)
    path_interp = decoder(z) + \
        tf.range(1/bins/2, 1, 1/bins)[:, None, None, None] * (x - decoder(z))

    # Cumulative trapezoidal integration, evaluate and sum each grid point
    integrand_entropy = tf.zeros([*decoder(z).shape[1:], z.shape[1]], dtype=tf.float32)
    integrand_aleatoric = tf.zeros([*decoder(z).shape[1:], z.shape[1]], dtype=tf.float32)
    integrand_entropy_interp = tf.zeros(x.shape[1:], dtype=tf.float32)
    integrand_aleatoric_interp = tf.zeros(x.shape[1:], dtype=tf.float32)

    # Loop and integrate numerically
    for i in tqdm.tqdm(range(bins)):
        entr, ale = d_entropy_d_z(path[i], decoder, model, sample)
        jacobian = get_jacobian(path[i], decoder, x.shape[0], x.shape[2], jacobian_splits)
        integrand_entropy += (entr[..., None] * jacobian) / bins
        integrand_aleatoric += (ale[..., None] * jacobian) / bins

        entr, ale = d_entropy_d_x(path_interp[i], model, sample)
        integrand_entropy_interp += entr / bins
        integrand_aleatoric_interp += ale / bins

    # LATENT PATH: Weight by and sum over latent value differences
    ig_entr = tf.tensordot(tf.squeeze(z - z_fid, axis=0), integrand_entropy,
                           axes=[[0], [3]])
    ig_ale = tf.tensordot(tf.squeeze(z - z_fid, axis=0), integrand_aleatoric,
                          axes=[[0], [3]])

    # INTERPOLATION PART: Weight by pixel value differences, sum over channels
    ig_entr += tf.squeeze(x - decoder(z), axis=0) * integrand_entropy_interp
    ig_ale += tf.squeeze(x - decoder(z), axis=0) * integrand_aleatoric_interp

    # Sum over channels
    ig_entr = tf.reduce_sum(ig_entr, axis=2)
    ig_ale = tf.reduce_sum(ig_ale, axis=2)
    ig_epi = ig_entr - ig_ale

    return ig_entr.numpy(), ig_ale.numpy(), ig_epi.numpy()
