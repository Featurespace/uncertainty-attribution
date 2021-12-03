import os
import tensorflow as tf
from uncertainty_library.data import read_format_data_CelebA
from uncertainty_library.models import build_keras_hq_images


# Train and save a model with uncertainty
# =============================================================================

ds_train, ds_valid = read_format_data_CelebA('img_align_celeba', category='Smiling', uniform_prob=0.1)

my_model = build_keras_hq_images(in_shape=(128, 128, 3))

epochs = 5
scheduler = tf.keras.optimizers.schedules.ExponentialDecay(1e-4, len(ds_train),
                                                           0.8, staircase=True)
opt = tf.keras.optimizers.Adam(learning_rate=scheduler)
cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# Compile the model
my_model.compile(optimizer=opt, loss=cce, metrics=[tf.keras.metrics.CategoricalAccuracy()])

# Train
fit = my_model.fit(ds_train, validation_data=ds_valid, epochs=epochs)

# Save the model - the predictions are too large to upload
os.makedirs('models', exist_ok=True)
my_model.save('models/classifier_smiling')
