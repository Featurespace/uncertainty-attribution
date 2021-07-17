import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def get_mc_predictions(model, dataset: tuple, samples=200):
    """Model predictions and uncertainty estimates using MC dropout,
       i.e. dropout applied at test time """

    ds_x = dataset[0]  # shape[dataset_length, W, H, C]
    ds_y = dataset[1]  # one-hot encoded, shape[dataset_length, num_classes]

    mc_predictions = np.zeros([samples, *ds_y.shape])
    for i in tqdm.tqdm(range(samples)):
        y_p = model.predict(ds_x, batch_size=128)
        mc_predictions[i, :] = y_p

    # Analyze distribution of probabilities
    predictive_dist = mc_predictions.mean(axis=0)
    entropy = -np.sum(predictive_dist * np.log(predictive_dist + 1e-100), axis=-1)
    aleatoric_unc_components = mc_predictions * np.log(mc_predictions + 1e-100)
    aleatoric_unc = np.mean(-np.nansum(aleatoric_unc_components, axis=-1), axis=0)
    epistemic_unc = entropy - aleatoric_unc

    # Build table of predictions and uncertainties
    summary = pd.DataFrame()
    summary['Label'] = np.argmax(ds_y, axis=1)
    summary['Pred'] = np.argmax(predictive_dist, axis=1)
    summary['Entropy'] = entropy
    summary['Aleatoric'] = aleatoric_unc
    summary['Epistemic'] = epistemic_unc

    return mc_predictions, summary


def get_entropy(x, model, samples=200, bs=32):
    # x assumed to be a batch of images
    ds = tf.data.Dataset.from_tensor_slices(x).batch(bs)
    entropies = []
    for batch in ds:
        B, W, H, C = batch.shape
        x_expanded = tf.repeat(batch[:, None, ...], samples, axis=1)
        x_expanded = tf.reshape(x_expanded, shape=(-1, W, H, C))
        preds = model(x_expanded)
        preds = tf.reshape(preds, shape=(B, samples, preds.shape[-1]))
        preds_mean = preds.numpy().mean(1)
        entropies.append(- np.sum(preds_mean * np.log(preds_mean + 1e-30), axis=-1))
    return np.concatenate(entropies)
