import os
import tensorflow as tf
from uncertainty_library.data import read_format_data
from uncertainty_library.models import (
    build_keras_images,
    vae_blocks_images,
    VAE
)


# Train and save a basic model with uncertainty
# =============================================================================

ds_train, ds_test = read_format_data('MNIST', horizontal_flip=False, augment=True,
                                     noise_prob=0.5, noise_level=0.1, uniform_prob=0.1)
my_model = build_keras_images(dropout_rate=0.5, in_shape=(28, 28, 1),
                              num_categories=10)

# Use binary cross entropy as loss
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Compile the model
my_model.compile(optimizer='adam', loss=cce,
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train
fit = my_model.fit(ds_train, validation_data=ds_test, epochs=10)

# Save the model - the predictions are too large to upload
os.makedirs('models', exist_ok=True)
my_model.save('models/model_MNIST')


# # Train and save a variational autoencoder
# # =============================================================================

ds_train, ds_test = read_format_data('MNIST', labelled=False, horizontal_flip=False, augment=True)

encoder, decoder = vae_blocks_images(in_shape=(28, 28, 1))
print(encoder.summary())
print(decoder.summary())
vae_model = VAE(encoder, decoder)

# Compile the model
vae_model.compile(optimizer='adam')

# Train
fit = vae_model.fit(ds_train, validation_data=ds_test, epochs=50)

# Save the encoder and decoder models
encoder.save('models/vae_MNIST/encoder')
decoder.save('models/vae_MNIST/decoder')
