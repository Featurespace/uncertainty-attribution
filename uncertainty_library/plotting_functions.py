import numpy as np
import matplotlib.pyplot as plt


def plot_that_pic(x, ig_entr, ig_ale, ig_epi, label=None, background='white', alpha=0.15):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.tight_layout(pad=0)

    if background == 'white':
        background_pic = np.ones_like(x)
    elif background == 'black':
        background_pic = np.zeros_like(x)
    else:
        raise ValueError(f'Invalid background flag : {background}. Should be "white" or "black"')

    # Images
    axes[0].imshow(x, cmap=plt.get_cmap('gray'))
    axes[0].axis('off')
    if label is None:
        axes[0].set_title('Original Image', fontsize=24)
    else:
        axes[0].set_title(f'{label.capitalize()}', fontsize=24)

    # ENTROPY -----------------------------------------------------------

    # Background pic
    axes[1].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[1].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = ig_entr.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / ig_entr.max()
    axes[1].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)
    axes[1].axis('off')

    # Add legend
    axes[1].set_title('Entropy Attribution Mask', fontsize=24)

    # ALEATORIC -----------------------------------------------------------

    # Background pic
    axes[2].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[2].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = ig_ale.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / max(ig_entr.max(), ig_ale.max())
    axes[2].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)
    axes[2].axis('off')

    # Add legend
    axes[2].set_title('Aleatoric Attribution Mask', fontsize=24)

    # EPISTEMIC -----------------------------------------------------------

    # Background pic
    axes[3].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[3].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = ig_epi.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / max(ig_entr.max(), ig_epi.max())
    axes[3].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)
    axes[3].axis('off')

    # Add legend
    axes[3].set_title('Epistemic Attribution Mask', fontsize=24)

    return fig


def plot_that_comparison(x, ig_entr, ig_entr_vanilla, entr_clue, entr_lime, entr_shap,
                         label=None, background='white', alpha=0.15):
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))
    fig.tight_layout(pad=0)

    if background == 'white':
        background_pic = np.ones_like(x)
    elif background == 'black':
        background_pic = np.zeros_like(x)
    else:
        raise ValueError(f'Invalid background flag : {background}. Should be "white" or "black"')

    # Images
    axes[0].imshow(x, cmap=plt.get_cmap('gray'))
    axes[0].axis('off')
    if label is None:
        axes[0].set_title('Original Image', fontsize=24)
    else:
        axes[0].set_title(f'{label.capitalize()}', fontsize=24)

    # IG LATENT -----------------------------------------------------------

    # Background pic
    axes[1].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[1].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = ig_entr.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / np.abs(ig_entr).max()
    axes[1].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)

    # Neg importances
    importances_neg = -ig_entr.copy()
    importances_neg[importances_neg < 0] = 0
    alpha_neg = importances_neg / np.abs(ig_entr).max()
    axes[1].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('summer'),
                   alpha=alpha_neg, vmin=0, vmax=1)
    axes[1].axis('off')

    # Add legend
    axes[1].set_title('Proposed Method', fontsize=24)

    # IG VANILLA -----------------------------------------------------------

    # Background pic
    axes[2].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[2].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = ig_entr_vanilla.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / np.abs(ig_entr_vanilla).max()
    axes[2].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)

    # Neg importances
    importances_neg = -ig_entr_vanilla.copy()
    importances_neg[importances_neg < 0] = 0
    alpha_neg = importances_neg / np.abs(ig_entr_vanilla).max()
    axes[2].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('summer'),
                   alpha=alpha_neg, vmin=0, vmax=1)
    axes[2].axis('off')

    # Add legend
    axes[2].set_title('Vanilla IG', fontsize=24)

    # CLUE -----------------------------------------------------------

    # Background pic
    axes[3].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[3].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = entr_clue.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / np.abs(entr_clue).max()
    axes[3].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)

    # Neg importances
    importances_neg = -entr_clue.copy()
    importances_neg[importances_neg < 0] = 0
    alpha_neg = importances_neg / np.abs(entr_clue).max()
    axes[3].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('summer'),
                   alpha=alpha_neg, vmin=0, vmax=1)
    axes[3].axis('off')

    # Add legend
    axes[3].set_title('CLUE', fontsize=24)

    # LIME -----------------------------------------------------------

    # Background pic
    axes[4].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[4].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = entr_lime.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / np.abs(entr_lime).max()
    axes[4].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)

    # Neg importances
    importances_neg = -entr_lime.copy()
    importances_neg[importances_neg < 0] = 0
    alpha_neg = importances_neg / np.abs(entr_lime).max()
    axes[4].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('summer'),
                   alpha=alpha_neg, vmin=0, vmax=1)
    axes[4].axis('off')

    # Add legend
    axes[4].set_title('LIME', fontsize=24)

    # SHAP -----------------------------------------------------------

    # Background pic
    axes[5].imshow(background_pic, cmap=plt.get_cmap('gray'), alpha=alpha)
    axes[5].imshow(x, cmap=plt.get_cmap('gray'), alpha=alpha)

    # Pos importances
    importances_pos = entr_shap.copy()
    importances_pos[importances_pos < 0] = 0
    alpha_pos = importances_pos / np.abs(entr_shap).max()
    axes[5].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('autumn'),
                   alpha=alpha_pos, vmin=0, vmax=1)

    # Neg importances
    importances_neg = -entr_shap.copy()
    importances_neg[importances_neg < 0] = 0
    alpha_neg = importances_neg / np.abs(entr_shap).max()
    axes[5].imshow(np.zeros(x.shape[:2]), cmap=plt.get_cmap('summer'),
                   alpha=alpha_neg, vmin=0, vmax=1)
    axes[5].axis('off')

    # Add legend
    axes[5].set_title('SHAP', fontsize=24)

    return fig