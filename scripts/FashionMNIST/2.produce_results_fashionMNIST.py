import os
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from uncertainty_library.mc_drop import get_mc_predictions
from uncertainty_library.data import (
    read_format_data,
    fashionMNIST_idx2label as idx2label,
    ds2numpy
)
from uncertainty_library.attribution_methods import get_importances
from uncertainty_library.plotting_functions import plot_that_pic


# Get the data
ds_train, ds_test = read_format_data('FashionMNIST', bs=10)
x_test, y_test = ds2numpy(ds_test, max_num_batches=200)

# Load the models
encoder = tf.keras.models.load_model('saved_models/vae_fashionMNIST/encoder')
decoder = tf.keras.models.load_model('saved_models/vae_fashionMNIST/decoder')
my_model = tf.keras.models.load_model('saved_models/model_fashionMNIST')

# Get predictions across the test data
mc_predictions, summary = get_mc_predictions(my_model, (x_test, y_test))
summary_selected = summary.loc[summary.Label == summary.Pred]


# Choose image and produce importances
# =============================================================================

indices = set(itertools.chain(
    summary_selected.sort_values(by="Epistemic").index[-300: -250],
    summary_selected.sort_values(by="Aleatoric").index[-300: -250]
))

os.makedirs('plots_fashionMNIST', exist_ok=True)
for idx in indices:
    x = x_test[idx]
    y = int(summary_selected.loc[idx].Label)
    label = idx2label[y]

    # Importances
    ig_entr, ig_ale, ig_epi = get_importances(x, my_model, encoder, decoder, known_class=y)

    # Plots
    fig = plot_that_pic(x, ig_entr, ig_ale, ig_epi, label=label,
                        background='black', alpha=0.3)
    fig.savefig(f'plots_fashionMNIST/{idx}.png',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
