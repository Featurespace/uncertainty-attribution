import os
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from uncertainty_library.mc_drop import get_mc_predictions
from uncertainty_library.data import read_format_data, ds2numpy
from uncertainty_library.attribution_methods import get_importances
from uncertainty_library.plotting_functions import plot_that_pic


# Get the data
ds_train, ds_test = read_format_data('MNIST', bs=10)
x_test, y_test = ds2numpy(ds_test, max_num_batches=200)

# Load the models
encoder = tf.keras.models.load_model('saved_models/vae_MNIST/encoder')
decoder = tf.keras.models.load_model('saved_models/vae_MNIST/decoder')
my_model = tf.keras.models.load_model('saved_models/model_MNIST')

# Get predictions across the test data
mc_predictions, summary = get_mc_predictions(my_model, (x_test, y_test))


# Choose images and produce importances
# =============================================================================

indices = set(itertools.chain(
    summary.sort_values(by="Entropy").index[-30:],
    summary.sort_values(by="Epistemic").index[-30:],
    summary.sort_values(by="Aleatoric").index[-30:]
))

os.makedirs('plots_MNIST', exist_ok=True)
for idx in indices:
    x = x_test[idx]
    y = int(summary.loc[idx].Label)

    # Importances
    ig_entr, ig_ale, ig_epi = get_importances(x, my_model, encoder, decoder, known_class=y)

    # Plots
    fig = plot_that_pic(1-x, ig_entr, ig_ale, ig_epi)
    fig.savefig(f'plots_MNIST/{idx}.png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
