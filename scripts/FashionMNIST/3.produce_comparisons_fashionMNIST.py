import os
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from uncertainty_library.data import (
    read_format_data,
    fashionMNIST_idx2label as idx2label,
)
from uncertainty_library.mc_drop import get_mc_predictions
from uncertainty_library.attribution_methods import (
    get_importances,
    get_importances_vanilla,
    get_importances_CLUE,
    get_importances_LIME,
    get_importances_SHAP,
)
from uncertainty_library.plotting_functions import plot_that_comparison


# Get the data
(x_train, y_train), (x_test, y_test) = read_format_data('FashionMNIST')

# Load the models
encoder = tf.keras.models.load_model('saved_models/vae_fashionMNIST/encoder')
decoder = tf.keras.models.load_model('saved_models/vae_fashionMNIST/decoder')
my_model = tf.keras.models.load_model('saved_models/model_fashionMNIST')

# Get predictions across the test data
mc_predictions, summary = get_mc_predictions(my_model, (x_test, y_test))
summary_selected = summary.loc[summary.Label == summary.Pred]


# Choose images and produce importances
# =============================================================================

indices = set(itertools.chain(
    summary_selected.sort_values(by="Epistemic").index[-300: -250],
    summary_selected.sort_values(by="Aleatoric").index[-300: -250]
))

os.makedirs('comparisons_fashionMNIST', exist_ok=True)
for idx in indices:
    x = x_test[idx]
    y = int(summary_selected.loc[idx].Label)
    label = idx2label[y]

    # Importances
    ig_entr, ig_ale, ig_epi = get_importances(x, my_model, encoder, decoder, known_class=y)

    ig_entr_vanilla, ig_ale_vanilla, ig_epi_vanilla = get_importances_vanilla(x, my_model)

    entr_clue = get_importances_CLUE(x, my_model, encoder, decoder)

    entr_lime = get_importances_LIME(x, my_model)

    entr_shap = get_importances_SHAP(x, my_model, x_train.numpy(), alpha=0.0005)

    fig = plot_that_comparison(x, ig_entr, ig_entr_vanilla, entr_clue,
                               entr_lime, entr_shap, label=label,
                               background='black', alpha=0.3)

    # Plots
    fig.savefig(f'comparisons_fashionMNIST/{idx}.png',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)
