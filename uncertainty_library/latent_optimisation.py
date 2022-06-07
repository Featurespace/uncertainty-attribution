""" Optimisation functions in latent space"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def get_latent_representation(
        image: tf.Tensor,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model):
    """Optimise the latent represantation of an image:
        1. Goal is to minimise the decoder reconstruction error

    Args:
        image (tf.Tensor): real image
        encoder (tf.keras.Model): encoder model
        decoder (tf.keras.Model): decoder model
    """
    # Set the starting point for optimization task
    z = tf.Variable(encoder(image[None, ])[0].numpy())

    # Define cost function, binary cross-entropy
    def loss_fn():
        x_entropy = tf.keras.losses.binary_crossentropy(image, decoder(z)[0])
        prior = tf.math.pow(z, 2)
        return tf.reduce_sum(x_entropy) + 0.5 * tf.reduce_sum(prior)

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z': z}

    # Minimise the loss!
    tol = tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=0.001)
    opt_method = tf.optimizers.Adam(learning_rate=0.01)
    tfp.math.minimize(loss_fn, num_steps=2000, optimizer=opt_method,
                      convergence_criterion=tol, trace_fn=trace_fn,
                      trainable_variables=[z])
    return z


def get_in_class_fiducial(
        x: tf.Tensor,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model,
        constrain_multiplier: np.float,
        known_class=None):
    """Optimises the latent representation to find a counterfactual fiducial
        1. Fiducial is optimised to:
            1.1. Minimise the entropy of prediction, and
            1.2. Be close to the original image
        2. Distance metric is cross-entropy in pixel space.

    Args:
        x (tf.Tensor): real image
        model (tf.keras.Model): classifier
        encoder (tf.keras.Model): encoder model
        decoder (tf.keras.Model): decoder model
        constrain_multiplier (np.float): constant multiplier to class penalty
        known_class (_type_, optional): Defaults to None.
    """

    # Determine the class prediction
    x_pred = model.predict(x[None, ])

    # Assign class to target for fiducial
    class_image = np.zeros(x_pred.shape[-1])
    if known_class is not None:
        class_image[known_class] = 1
    else:
        print(f"Inferred class is {np.argmax(x_pred)}")
        class_image[np.argmax(x_pred)] = 1

    # Set the starting point for optimization task
    z_fid = tf.Variable(encoder(x[None, ])[0].numpy())

    # Multiplier factor for loss on class constrain
    class_scale = constrain_multiplier * tf.reduce_prod(x.shape[:2]).numpy()

    # Define stochastic cost function, with binary cross-entropy
    def sto_loss_fn():
        decoded_fid = decoder(z_fid)
        pred_fid = model(decoded_fid)[0]

        # Class term
        class_term = class_scale * tf.keras.losses.binary_crossentropy(
            class_image, pred_fid)

        # Reconstruction term
        reconstruction_loss = tf.keras.losses.binary_crossentropy(
            x, decoded_fid[0])
        reconstruction_loss = tf.reduce_sum(reconstruction_loss)

        # Prior
        prior = tf.math.pow(z_fid, 2)
        prior = 0.5 * tf.reduce_sum(prior)

        return reconstruction_loss + class_term + prior

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z_fid': z_fid}

    # Minimise the loss!
    opt_met = tf.optimizers.Adam(learning_rate=0.01)
    tol = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)
    tfp.math.minimize(sto_loss_fn, num_steps=2000, optimizer=opt_met,
                      convergence_criterion=tol, trace_fn=trace_fn,
                      trainable_variables=[z_fid])
    return z_fid


def get_clue_counterfactual(
        x: tf.Tensor,
        model: tf.keras.Model,
        encoder: tf.keras.Model,
        decoder: tf.keras.Model):
    """Optimises the latent representation to find a counterfactual fiducial
        1. Follows CLUE guidelines

    Args:
        x (tf.Tensor): real image
        model (tf.keras.Model): classifier
        encoder (tf.keras.Model): encoder model
        decoder (tf.keras.Model): decoder model
    """
    # Set the starting point for optimization task
    z_fid = tf.Variable(encoder(x[None, ])[0].numpy())

    # Define stochastic cost function
    def sto_loss_fn():
        decoded_fid = decoder(z_fid)
        pred_fid = model(decoded_fid)[0]

        # Reconstruction term
        reconstruction_loss = tf.reduce_sum(tf.abs(x - decoded_fid[0])) / x.shape[0] / x.shape[1]

        # Entropy of CLUE
        entropy = - tf.experimental.numpy.nansum(pred_fid * tf.math.log(pred_fid))

        return reconstruction_loss + entropy

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z_fid': z_fid}

    # Minimise the loss!
    opt_met = tf.optimizers.Adam(learning_rate=0.1)
    tol = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)
    tfp.math.minimize(sto_loss_fn, num_steps=1000, optimizer=opt_met,
                      convergence_criterion=tol, trace_fn=trace_fn,
                      trainable_variables=[z_fid])

    return z_fid
