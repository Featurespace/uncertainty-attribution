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

(x_train, y_train), (x_test, y_test) = read_format_data('MNIST')
my_model = build_keras_images(dropout_rate=0.5, in_shape=(28, 28, 1),
                              num_categories=10)

# Use binary cross entropy as loss
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Compile the model
my_model.compile(optimizer='adam', loss=cce,
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train
fit = my_model.fit(x_train, y_train, validation_data=(x_test, y_test),
                   epochs=10, batch_size=32)

# Save the model - the predictions are too large to upload
os.makedirs('saved_models', exist_ok=True)
my_model.save('saved_models/model_MNIST')


# Train and save a variational autoencoder
# =============================================================================

encoder, decoder = vae_blocks_images(in_shape=(28, 28, 1))
print(encoder.summary())
print(decoder.summary())
vae_model = VAE(encoder, decoder)

# Compile the model
vae_model.compile(optimizer='adam')

# Train
fit = vae_model.fit(x_train, epochs=50, batch_size=32)

# Save the encoder and decoder models
encoder.save('saved_models/vae_MNIST/encoder')
decoder.save('saved_models/vae_MNIST/decoder')
