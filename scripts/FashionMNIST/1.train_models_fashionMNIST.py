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

(x_train, y_train), (x_test, y_test) = read_format_data('FashionMNIST')

my_model = build_keras_images(dropout_rate=0.4, in_shape=(28, 28, 1),
                              num_categories=10)

# Use binary cross entropy as loss
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=3e-4)

# Compile the model
my_model.compile(optimizer=opt, loss=cce,
                 metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train
fit = my_model.fit(x_train, y_train, validation_data=(x_test, y_test),
                   epochs=30, batch_size=32)

# Save the model - the predictions are too large to upload
os.makedirs('saved_models', exist_ok=True)
my_model.save('saved_models/model_fashionMNIST')


# Train and save a variational autoencoder
# =============================================================================

encoder, decoder = vae_blocks_images(in_shape=(28, 28, 1))
print(encoder.summary())
print(decoder.summary())
vae_model = VAE(encoder, decoder)

# Compile the model
epochs = 80
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-3, len(x_train), 0.97, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=scheduler)
vae_model.compile(optimizer=opt)

# Train
fit = vae_model.fit(x_train, epochs=epochs, batch_size=32)

# Save the encoder and decoder models
encoder.save('saved_models/vae_fashionMNIST/encoder')
decoder.save('saved_models/vae_fashionMNIST/decoder')
