import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from uncertainty_library.mc_drop import get_mc_predictions
from uncertainty_library.data import read_format_data_CelebA
from uncertainty_library.attribution_methods import (
    get_importances,
    get_importances_vanilla,
    get_importances_CLUE,
    get_importances_LIME,
    get_importances_SHAP,
)
from uncertainty_library.plotting_functions import plot_that_comparison,


def ds2numpy(ds, max_num_batches):
    x, y = [], []
    for batch in ds.take(max_num_batches):
        x.append(batch[0].numpy())
        y.append(batch[1].numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y


train_ds, valid_ds = read_format_data_CelebA('img_align_celeba', category='Smiling')
x_train, y_train = ds2numpy(train_ds, max_num_batches=200)
x_valid, y_valid = ds2numpy(valid_ds, max_num_batches=200)
idx2label = ['neg', 'pos']


# Load the models
encoder = tf.keras.models.load_model('saved_models/dfc-vae-cnn/encoder')
decoder = tf.keras.models.load_model('saved_models/dfc-vae-cnn/decoder')
my_model = tf.keras.models.load_model('saved_models/classifier_smiling')


# Get predictions across the test data
mc_predictions, summary = get_mc_predictions(my_model, (x_valid, y_valid))
summary_selected = summary.loc[summary.Label == summary.Pred]
summary_selected = summary_selected[
    (summary_selected.Entropy > 0.35) & (summary_selected.Entropy < 0.45)
]


# Choose image and produce importances
# =============================================================================

os.makedirs('plots_comparisons', exist_ok=True)
for idx in summary_selected.sort_values(by="Epistemic").index[:50]:
    x = x_valid[idx]
    y = int(summary_selected.loc[idx].Label)
    label = idx2label[y]

    # Importances
    ig_entr, ig_ale, ig_epi = get_importances(x, my_model, encoder, decoder,
                                              known_class=y, jacobian_splits=16)

    ig_entr_vanilla, ig_ale_vanilla, ig_epi_vanilla = get_importances_vanilla(x, my_model)

    entr_clue = get_importances_CLUE(x, my_model, encoder, decoder)

    entr_lime = get_importances_LIME(x, my_model, num_perturb=5000, alpha=7e-4, bs=4)

    entr_shap = get_importances_SHAP(x, my_model, x_train, num_perturb=50000,
                                     alpha=1e-4, bs=4)

    # Plots
    fig = plot_that_comparison(x, ig_entr, ig_entr_vanilla, entr_clue,
                               entr_lime, entr_shap, label=label, alpha=0.5)
    fig.savefig(f'plots_comparisons/{idx}.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
