
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def get_latent_representation(image, encoder, decoder):
    """Optimise the latent represantation of an image to
       minimise the decoder reconstruction error"""

    # Set the starting point for optimization task
    z = tf.Variable(encoder(image[None, ])[0].numpy())

    # Define cost function, binary cross-entropy
    def loss_fn():
        x_entropy = tf.keras.losses.binary_crossentropy(image, decoder(z)[0])
        return tf.reduce_sum(x_entropy)

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z': z}

    # Minimise the loss!
    tol = tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=0.0001)
    opt_method = tf.optimizers.Adam(learning_rate=0.01)
    tfp.math.minimize(loss_fn, num_steps=5000, optimizer=opt_method,
                      convergence_criterion=tol, trace_fn=trace_fn,
                      trainable_variables=[z])
    return z


def get_in_class_fiducial(x, model, encoder, decoder, known_class=None):
    """Optimises the latent representation to find a counterfactual fiducial.
       The fiducial is optimised to minimise the entropy of prediction and
       at the same time be close to the original image, with the distance
       metric being cross-entropy in pixel space."""

    # Determine the class prediction
    mc = model.predict(tf.repeat(x[None, :], 1000, axis=0)).mean(0)

    # Assign class to target for fiducial
    class_image = np.zeros(mc.shape[-1])
    if known_class is not None:
        class_image[known_class] = 1
    else:
        print(f"Inferred class is {np.argmax(mc)}")
        class_image[np.argmax(mc)] = 1

    # Set the starting point for optimization task
    z_fid = tf.Variable(encoder(x[tf.newaxis, ])[0].numpy())

    # Define stochastic cost function, with binary cross-entropy
    def sto_loss_fn():
        decoded_fid = decoder(z_fid)
        mc_fid = tf.reduce_mean(model(tf.repeat(decoded_fid, 100, axis=0)),
                                axis=0)

        # Class term
        class_term = 1e2 * tf.keras.losses.binary_crossentropy(class_image, mc_fid)

        # Reconstruction term
        reconstruction_loss = tf.keras.losses.binary_crossentropy(x, decoded_fid[0])
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)

        return reconstruction_loss + class_term

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z_fid': z_fid}

    # Minimise the loss!
    opt_met = tf.optimizers.Adam(learning_rate=0.01)
    tfp.math.minimize(sto_loss_fn, num_steps=3000, optimizer=opt_met,
                      trace_fn=trace_fn, trainable_variables=[z_fid])
    return z_fid


@tf.function
def d_entropy_d_z(z, decoder, model, sample):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(z)

        # Decoded latent space
        z_decoded = decoder(z[None, ...])[0]

        # MC samples of scores
        scores = model(tf.repeat(z_decoded[None, ...], sample, axis=0))

        # Compute the entropy of expected score
        score_m = tf.reduce_mean(scores, axis=0)
        entr = - tf.reduce_sum(score_m * tf.math.log(score_m + 1e-30))

        # Compute expected entropy across scores
        entropies = tf.reduce_sum(scores * tf.math.log(scores + 1e-30), axis=1)
        entr_ale = - tf.reduce_mean(entropies)

    # Derivative of entropy wrt decoded image
    d_entr_d_deco = tape.gradient(entr, z_decoded)
    d_ale_d_deco = tape.gradient(entr_ale, z_decoded)

    return d_entr_d_deco, d_ale_d_deco


@tf.function
def get_jacobian_slice(z, begin, size, decoder):
    with tf.GradientTape() as tape:
        tape.watch(z)

        # Decoded latent space
        z_decoded = decoder(z[None, ...])[0]
        z_decoded_slice = tf.slice(z_decoded, begin=begin, size=size)

    # Derivative of decoded image wrt latent vector
    jacobian_slice = tape.jacobian(z_decoded_slice, z)
    return jacobian_slice


def get_jacobian(z, decoder, img_size=128, num_channels=3, iters=16):
    """Computes the jacobian in chunks which fit in GPU memory"""

    jacobian_slices = []
    for i in range(iters):
        begin = tf.constant([i * (img_size // iters), 0, 0], dtype=tf.int32)
        size = tf.constant([img_size // iters, img_size, num_channels], dtype=tf.int32)

        jacobian_slice = get_jacobian_slice(z, begin, size, decoder)
        jacobian_slices.append(jacobian_slice)
    return tf.concat(jacobian_slices, axis=0)


def get_clue_counterfactual(x, model, encoder, decoder):
    # Set the starting point for optimization task
    z_fid = tf.Variable(encoder(x[None, ])[0].numpy())

    # Define stochastic cost function
    def sto_loss_fn():
        decoded_fid = decoder(z_fid)
        mc_fid = tf.reduce_mean(model(tf.repeat(decoded_fid, 100, axis=0)),
                                axis=0)

        # Reconstruction term
        reconstruction_loss = 0.01 * tf.reduce_sum(tf.abs(x - decoded_fid[0]))

        # Entropy of CLUE
        entropy = -tf.experimental.numpy.nansum(mc_fid * tf.math.log(mc_fid))

        return reconstruction_loss + entropy

    # Define trace function
    def trace_fn(traceable_quantities):
        return {'loss': traceable_quantities.loss, 'z_fid': z_fid}

    # Minimise the loss!
    opt_met = tf.optimizers.Adam(learning_rate=0.1)
    tfp.math.minimize(sto_loss_fn, num_steps=1000, optimizer=opt_met,
                      trace_fn=trace_fn, trainable_variables=[z_fid])

    return z_fid
