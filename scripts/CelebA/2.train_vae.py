import os
import tensorflow as tf
from uncertainty_library.data import read_format_data_CelebA
from uncertainty_library.models import vae_blocks_hq_images, VAE


# Read the data
# =============================================================================

ds_train, ds_valid = read_format_data_CelebA('img_align_celeba')

encoder, decoder = vae_blocks_hq_images(in_shape=(128, 128, 3))
vae_model = VAE(encoder, decoder, do_perceptual_loss=True)


# Train and save the model - which includes encoder and decoder
# =============================================================================

# Compile the model
epochs = 100
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(5e-4, len(ds_train),
                                                           0.98, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=scheduler)
vae_model.compile(optimizer=opt)

# Train
fit = vae_model.fit(ds_train, validation_data=ds_valid, epochs=epochs, verbose=2)

# Save the encoder and decoder models
os.makedirs('saved_models', exist_ok=True)
encoder.save('saved_models/dfc-vae-cnn/encoder')
decoder.save('saved_models/dfc-vae-cnn/decoder')
