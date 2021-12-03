from typing import Union

import numpy as np
import tensorflow as tf
from scipy.ndimage import gaussian_filter


def _get_avg_bin_vals(y: np.ndarray, num_bins: int):
    bin_right_vals, bin_sep = np.linspace(0.0, 100.0, num_bins + 1, retstep=True)
    bin_centres = np.concatenate(([0.0], bin_right_vals[1:] - bin_sep / 2, [100.0]))

    segment_ids = np.linspace(0.0, num_bins, tf.shape(y)[0]).astype(np.int64)
    segment_ids[1:] += 1
    avg_per_bin = tf.math.segment_mean(y, segment_ids).numpy()

    return avg_per_bin, bin_centres


def _break_degeneracy(attributions: tf.Tensor) -> tf.Tensor:
    sorted_attributions = tf.sort(attributions)
    differences = sorted_attributions[1:] - sorted_attributions[:-1]
    if tf.reduce_all(differences == 0.0):
        return tf.random.uniform(attributions.shape, 0.0, 1.0)
    smallest_difference = tf.reduce_min(tf.abs(differences[differences != 0.0]))
    attributions = attributions + tf.random.uniform(attributions.shape, - smallest_difference, smallest_difference)
    return attributions


def pic_vs_px_count(img: Union[tf.Tensor, np.ndarray], attributions: Union[tf.Tensor, np.ndarray],
                    model, blur: int, metric: str, num_bins: int = 100, sort_direction='ASCENDING'):
    """
    Produces a performance information curve (PIC), which shows how model performance (either softmax
    or entropy) changes as the pixels of the image are revealed in the order given by the attribution
    map (in the direction specified by `sort_direction`). The background image is a blurred version of the
    original image (using gaussian blurring with sigma=`blur`).

    The beginning of x axis corresponds to a fully blurred image, while the end corresponds to the
    original image. The y axis can either be softmax of the class with maximum score, or entropy. We
    generally consider attributions to be better if the AUC under the softmax/entropy information curve
    is bigger/smaller (respectively).
    """
    img = tf.convert_to_tensor(img)
    attributions = tf.convert_to_tensor(attributions, dtype=tf.float32)
    attributions = _break_degeneracy(attributions)

    with tf.device('CPU'):
        img_blurred = tf.convert_to_tensor(gaussian_filter(img, sigma=(blur, blur, 0)))
        sorted_attributions = tf.sort(tf.reshape(attributions, [-1]), direction=sort_direction)
        occluded_images = tf.where(
            attributions[None, ..., None] >= sorted_attributions[..., None, None, None],
            img_blurred[None, ...],
            img[None, ...]
        )

    preds = model.predict(occluded_images)
    if metric == 'softmax':
        class_idx = np.argmax(preds[-1])
        vals = preds[..., class_idx]
        avg_bin_vals, bins = _get_avg_bin_vals(vals, num_bins)
    elif metric == 'entropy':
        vals = - np.sum(preds * np.log(preds), axis=-1)
        avg_bin_vals, bins = _get_avg_bin_vals(vals, num_bins)
    else:
        raise ValueError("Wrong metric type")

    return avg_bin_vals, bins, {'occluded_images': occluded_images, 'predictions': preds}
