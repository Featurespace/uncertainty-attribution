""" Keras models used within experiments in this repository"""

import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input


# Basic Keras CNN models with dropout
# =============================================================================

def build_keras_images(
        in_shape: tuple,
        dropout_rate: float,
        num_categories: int):
    """Basic classification model of low resolution images.

    Args:
        in_shape (tuple): input shape
        dropout_rate (float): dropout rate
        num_categories (int): number of output categories
    """
    input_tensor = tf.keras.layers.Input(shape=in_shape, name='input_layer')

    out = tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    out = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), activation='relu')(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)

    # Flatten and proceed with MLP and dropout
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(128, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    out = tf.keras.layers.Dense(num_categories, activation='softmax')(out)

    model = tf.keras.Model(inputs=input_tensor, outputs=out)
    print(model.summary())

    return model


def build_keras_hq_images(
        in_shape: tuple,
        conv_drop_rate: float = 0.2,
        cls_drop_rate: float = 0.4):
    """Classification model for higher resolution images.

    Args:
        in_shape (tuple): input shape
        conv_drop_rate (float, optional): Conv dropout rate. Defaults to 0.2.
        cls_drop_rate (float, optional): Cls dropout rate. Defaults to 0.4.
    """
    input_tensor = tf.keras.layers.Input(shape=in_shape)

    out = tf.keras.layers.Conv2D(32, 3, 1, padding='same')(input_tensor)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(64, 3, 1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(128, 3, 1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Conv2D(256, 3, 1, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dropout(conv_drop_rate)(out)
    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dropout(cls_drop_rate)(out)
    out = tf.keras.layers.Dense(2, activation='softmax')(out)

    model = tf.keras.Model(input_tensor, out)
    print(model.summary())

    return model


# Definition of encoder and decoder blocks for VAE
# =============================================================================

def vae_blocks_images(in_shape: tuple):
    """Create encoder/decoder models for  experiments on low resolution images.

    Args:
        in_shape (tuple): input shape
    """
    # ENCODER
    # =========================================================================
    input_tensor = tf.keras.layers.Input(shape=in_shape)

    out = tf.keras.layers.Conv2D(
        32, 3, 2, padding='same', activation='relu')(input_tensor)
    out = tf.keras.layers.Conv2D(
        64, 3, 2, padding='same', activation='relu')(out)
    size = out.shape[1]

    flat = tf.keras.layers.Flatten()(out)
    dense = tf.keras.layers.Dense(128, activation='relu')(flat)
    mu = tf.keras.layers.Dense(32)(dense)
    log_sigma_2 = tf.keras.layers.Dense(32)(dense)

    # Sample a random latent vector from gaussian distribution
    eps = tf.random.normal(shape=tf.shape(mu))
    out = eps * tf.exp(log_sigma_2 * .5) + mu
    encoder = tf.keras.Model(input_tensor, [mu, log_sigma_2, out])
    print(encoder.summary())

    # DECODER
    # =========================================================================
    input_tensor = tf.keras.layers.Input(shape=(32,))

    out = tf.keras.layers.Dense(
        size ** 2 * 64, activation="relu")(input_tensor)
    out = tf.keras.layers.Reshape((size, size, 64))(out)
    out = tf.keras.layers.Conv2DTranspose(
        64, 3, 2, padding="same", activation="relu")(out)
    out = tf.keras.layers.Conv2DTranspose(
        32, 3, 2, padding="same", activation="relu")(out)
    out = tf.keras.layers.Conv2DTranspose(
        1, 3, 1, padding="same", activation="sigmoid")(out)

    decoder = tf.keras.Model(input_tensor, out)
    print(decoder.summary())

    return encoder, decoder


class UpsampleConvLayer(tf.Module):
    """Definition of an upsampling convolutional layer

    Args:
        tf (_type_): _description_
    """
    def __init__(self, out_channels, kernel_size, stride=1, upsample=2):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=kernel_size,
            strides=stride,
            padding='same')

    def __call__(self, x):
        _, W, H, C = x.shape[0], x.shape[1], x.shape[2], x.shape[3]

        # Upsample
        out = tf.expand_dims(tf.expand_dims(x, 2), 4)
        out = tf.tile(out, [1, 1, self.upsample, 1, self.upsample, 1])
        out = tf.reshape(out, [-1, W * self.upsample, H * self.upsample, C])

        out = self.conv(out)
        return out


def vae_blocks_hq_images(in_shape: tuple):
    """Create encoder/decoder models for experiments on high resolution images.

    Args:
        in_shape (tuple): input shape
    """
    # ENCODER
    # =========================================================================
    input_ = tf.keras.layers.Input(shape=in_shape)

    # ENCODER
    out = tf.keras.layers.Conv2D(32, 3, 2, padding='same')(input_)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = tf.keras.layers.Conv2D(64, 3, 2, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = tf.keras.layers.Conv2D(256, 3, 2, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = tf.keras.layers.Conv2D(512, 3, 2, padding='same')(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    flat = tf.keras.layers.Flatten()(out)
    mu = tf.keras.layers.Dense(256)(flat)
    log_sigma_2 = tf.keras.layers.Dense(256)(flat)

    # Sample latent vector
    eps = tf.random.normal(shape=tf.shape(mu))
    output = eps * tf.exp(log_sigma_2 * .5) + mu
    encoder = tf.keras.Model(input_, [mu, log_sigma_2, output])
    print(encoder.summary())

    # DECODER
    size = out.shape[1]
    input_latent = tf.keras.layers.Input(shape=(256,))
    out = tf.keras.layers.Dense(size ** 2 * 512)(input_latent)
    out = tf.keras.layers.Reshape((size, size, 512))(out)

    out = UpsampleConvLayer(256, 3, stride=1, upsample=2)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = UpsampleConvLayer(128, 3, stride=1, upsample=2)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = UpsampleConvLayer(64, 3, stride=1, upsample=2)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = UpsampleConvLayer(32, 3, stride=1, upsample=2)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = UpsampleConvLayer(3, 3, stride=1, upsample=2)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.LeakyReLU()(out)

    out = tf.keras.layers.Conv2D(3, 3, 1, padding='same')(out)
    out = tf.keras.activations.sigmoid(out)
    decoder = tf.keras.Model(input_latent, out)
    print(decoder.summary())

    return encoder, decoder


# VAE model as a class with custom train_step
# =============================================================================

class VAE(tf.keras.Model):
    """Implementation of a Variational Autoencoder class"""
    def __init__(self, encoder, decoder, do_perceptual_loss=False, **kwargs):
        super(VAE, self).__init__(**kwargs)

        # Basic encode decode blocks
        self.encoder = encoder
        self.decoder = decoder

        # Define the loss trackers
        self.loss_tr = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tr = tf.keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tr = tf.keras.metrics.Mean(name="kl_loss")

        if do_perceptual_loss:
            self.reconstruction_loss_fn = self.get_perceptual_loss()
        else:
            self.reconstruction_loss_fn = self.get_crossentropy_pixel_loss()

    def get_crossentropy_pixel_loss(self):
        def loss_fn(image, decoded):
            return tf.reduce_mean(tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(image, decoded),
                axis=(1, 2)
            ))
        return loss_fn

    def get_perceptual_loss(self):
        # https://arxiv.org/pdf/1610.00291.pdf
        vgg_base = VGG19(
            input_shape=(128, 128, 3), weights='imagenet', include_top=False)
        loss_model = tf.keras.models.Model(
            inputs=vgg_base.input,
            outputs=[
                vgg_base.get_layer('block1_conv1').output,
                vgg_base.get_layer('block2_conv1').output,
                vgg_base.get_layer('block3_conv1').output
            ]

        )
        for layer in loss_model.layers:
            layer.trainable = False

        def loss_fn(image, decoded):
            image_ = preprocess_input(image * 255.0)
            decoded_ = preprocess_input(decoded * 255.0)
            image_features = loss_model(image_)
            decoded_features = loss_model(decoded_)
            losses = []
            for img_feats, dec_feats in zip(image_features, decoded_features):
                losses.append(
                    0.5 * tf.reduce_mean((img_feats - dec_feats) ** 2)
                )
            return tf.math.add_n(losses)

        return loss_fn

    @property
    def metrics(self):
        return [self.loss_tr, self.reconstruction_loss_tr, self.kl_loss_tr]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            # Encode and decode image
            mu, log_sigma_2, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Loss on image reconstruction
            reconstruction_loss = self.reconstruction_loss_fn(
                data, reconstruction)

            # Loss on KL generative regularization term
            kl_loss = \
                -0.5 * (1 + log_sigma_2 - tf.square(mu) - tf.exp(log_sigma_2))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # Add the total loss
            total_loss = reconstruction_loss + kl_loss

        # Gradient descent step
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update state of loss trackers
        self.loss_tr.update_state(total_loss)
        self.reconstruction_loss_tr.update_state(reconstruction_loss)
        self.kl_loss_tr.update_state(kl_loss)

        return {
            "loss": self.loss_tr.result(),
            "reconstruction_loss": self.reconstruction_loss_tr.result(),
            "kl_loss": self.kl_loss_tr.result(),
        }

    def test_step(self, data):
        # Encode and decode image
        mu, log_sigma_2, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Loss on image reconstruction
        reconstruction_loss = self.reconstruction_loss_fn(data, reconstruction)

        # Loss on KL generative regularization term
        kl_loss = \
            -0.5 * (1 + log_sigma_2 - tf.square(mu) - tf.exp(log_sigma_2))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

        # Add the total loss
        total_loss = reconstruction_loss + kl_loss

        # Update state of loss trackers
        self.loss_tr.update_state(total_loss)
        self.reconstruction_loss_tr.update_state(reconstruction_loss)
        self.kl_loss_tr.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}
