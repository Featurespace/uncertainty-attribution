"""Support functions to pre-process and format data"""

import os
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


# Data reading and formatting
# =============================================================================

def replace_subset_with_uniform_images(sampling_prob: float) -> Callable:
    """
    Returns a transformation to be used with tf.data.Dataset.map:
        1. Sample portion of input images -> swaps for uniform, random color.
        2. Swaps labels, signifies an unknown class.

    Args:
        sampling_prob (float): Range [0.0, 1.0], fraction of images and labels.
    """
    def transform(images, labels):
        shape_ = tf.shape(images)
        random_mask = tf.random.uniform(
            shape=[shape_[0]], minval=0.0, maxval=1.0) < sampling_prob
        random_colors = tf.random.uniform(
            shape=[shape_[0], 1, 1, shape_[3]], minval=0.0, maxval=1.0)
        images = tf.where(
            random_mask[..., None, None, None],
            random_colors * tf.ones_like(images),
            images
        )
        labels = tf.where(
            random_mask[..., None],
            tf.ones_like(
                labels) / tf.cast(tf.shape(labels)[-1], dtype=tf.float32),
            labels
        )
        return images, labels

    return transform


def random_noise(noise_prob: float, noise_level: float) -> Callable:
    """
    Returns a transformation to be used with tf.data.Dataset.map:
        1. Adds random noise to input images.

    Args:
        noise_prob (float): range [0.0, 1.0] - fraction of images
        noise_level (float): range [0.0, 1.0] - the amount of noise to add.
    """
    def transform(images):
        shape_ = tf.shape(images)
        random_mask = tf.random.uniform(
            shape=[shape_[0], 1, 1, shape_[3]],
            minval=0.0,
            maxval=1.0) < noise_prob
        images = tf.where(
            random_mask,
            images + tf.random.normal(shape_, stddev=noise_level),
            images
        )
        return images
    return transform


def _preprocess_dataset(
        ds: tf.data.Dataset,
        labelled: bool,
        augment: bool,
        horizontal_flip: bool,
        noise_prob: float,
        noise_level: float,
        uniform_prob: float):
    """Data preprocessing function for MNIST or FashionMNIST datasets.

    Args:
        ds (tf.data.Dataset): tensorflow dataser
        labelled (bool): labelled data?
        augment (bool): augment data?
        horizontal_flip (bool): flipe data?
        noise_prob (float): add random noise?
        noise_level (float): noise quantity
        uniform_prob (float): prob of replacing image with uniform background
    """
    if labelled:
        def decorator(func):
            def wrapped(image, label):
                return func(image), label
            return wrapped
    else:
        def decorator(func):
            return func

    if augment:
        augmentations = [
            preprocessing.RandomRotation(factor=0.05),
            preprocessing.RandomTranslation(
                height_factor=0.1, width_factor=0.1),
        ]
        if horizontal_flip:
            augmentations.append(preprocessing.RandomFlip(mode='horizontal'))
        augmentation = tf.keras.models.Sequential(
            augmentations, name='augmentation')
        ds = ds.map(
            decorator(augmentation), num_parallel_calls=tf.data.AUTOTUNE)

    if noise_prob > 0.0:
        ds = ds.map(
            decorator(random_noise(noise_prob, noise_level)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    ds = ds.map(decorator(lambda image: tf.clip_by_value(image, 0.0, 1.0)))

    if labelled:
        ds = ds.map(
            lambda image,
            label: (
                image,
                tf.squeeze(tf.one_hot(tf.cast(label, tf.int32), 10))
            )
        )
        if uniform_prob > 0.0:
            ds = ds.map(
                replace_subset_with_uniform_images(uniform_prob),
                num_parallel_calls=tf.data.AUTOTUNE
            )
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def read_format_data(
        dataset: str,
        labelled=True,
        bs=64,
        augment=True,
        horizontal_flip=False,
        noise_prob=0.0,
        noise_level=0.0,
        uniform_prob=0.0):
    """Primary procedure to leat and format input data.

    Args:
        dataset (str): path to input data
        labelled (bool, optional): Defaults to True.
        bs (int, optional): batch size. Defaults to 64.
        augment (bool, optional): Defaults to True.
        horizontal_flip (bool, optional): Defaults to False.
        noise_prob (float, optional): Defaults to 0.0.
        noise_level (float, optional): Defaults to 0.0.
        uniform_prob (float, optional): Defaults to 0.0.
    """
    if dataset == 'MNIST':
        ds = tf.keras.datasets.mnist
    elif dataset == 'FashionMNIST':
        ds = tf.keras.datasets.fashion_mnist
    else:
        raise ValueError(
            "`dataset` should be either `MNIST` or `FashionMNIST`, "
            f"but {dataset} ws provided")

    # Load and scale data
    (x_train, y_train), (x_test, y_test) = ds.load_data()

    x_train = np.expand_dims(x_train, 3) / 255.0
    x_test = np.expand_dims(x_test, 3) / 255.0

    # Create tensors
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
    if labelled:
        ds_train = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .batch(bs, drop_remainder=True)
                .shuffle(buffer_size=bs * 10, seed=7))
        ds_test = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(bs, drop_remainder=True)
    else:
        ds_train = (
            tf.data.Dataset.from_tensor_slices(x_train)
                .batch(bs, drop_remainder=True)
                .shuffle(buffer_size=bs * 10, seed=7))
        ds_test = (
            tf.data.Dataset.from_tensor_slices(x_test)
                .batch(bs, drop_remainder=True))

    ds_train = _preprocess_dataset(
        ds_train,
        labelled,
        augment,
        horizontal_flip,
        noise_prob,
        noise_level,
        uniform_prob)
    ds_test = _preprocess_dataset(
        ds_test,
        labelled,
        augment=False,
        horizontal_flip=False,
        noise_prob=0.0,
        noise_level=0.0,
        uniform_prob=0.0)

    return ds_train, ds_test


# Functions specific to CelebA data
# =============================================================================

fashionMNIST_idx2label = [
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
]


def get_labels_CelebA(folder: str, category):
    """Extract CelebA labels

    Args:
        folder (str): path to images
        category (_type_): _description_
    """
    with open('scripts/CelebA/list_attr_celeba.txt', 'r') as attr_file:
        next(attr_file)  # skip first dummy line
        attributes_names = next(attr_file).split()
        attributes_list = []
        file_names = []
        for line in attr_file:
            file_name, *attributes = line.split()
            file_names.append(file_name)
            attributes_list.append(attributes)

    attributes_dict = {}
    for file_name, attributes in zip(file_names, attributes_list):
        attributes_dict.update(
            {file_name: [int(attr == '1') for attr in attributes]})

    file_list = sorted(os.listdir(folder))
    category_idx = attributes_names.index(category)
    labels = []
    for file_name in file_list:
        label = attributes_dict[file_name][category_idx]
        labels.append(label)

    return labels


def _preprocess_dataset_CelebA(
        ds: tf.data.Dataset,
        augment=True,
        is_labelled=True,
        uniform_prob=0.0):
    """Pre-processing of CelebA images

    Args:
        ds (tf.data.Dataset): input dataset
        augment (bool, optional): Defaults to True.
        is_labelled (bool, optional): Defaults to True.
        uniform_prob (float, optional): Defaults to 0.0.
    """
    if is_labelled:
        def decorator(func):
            def wrapped(image, label):
                return func(image), label
            return wrapped
    else:
        def decorator(func):
            return func

    ds = ds.map(decorator(lambda image: image[:, 27: 205, :, :]))
    if augment:
        augmentation = tf.keras.models.Sequential(
            [
                preprocessing.RandomRotation(factor=0.05),
                preprocessing.RandomTranslation(
                    height_factor=0.1, width_factor=0.1),
                preprocessing.RandomFlip(mode='horizontal'),
            ],
            name='augmentation'
        )
        ds = ds.map(
            decorator(augmentation), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(decorator(lambda image: image[:, 25: -25, 25: -25, :]))
    ds = ds.map(
        decorator(lambda image: tf.clip_by_value(image, 0.0, 255.0) / 255.0))
    if is_labelled:
        ds = ds.map(
            lambda image,
            label: (image, tf.squeeze(tf.one_hot(tf.cast(label, tf.int32), 2)))
        )
        if uniform_prob:
            ds = ds.map(replace_subset_with_uniform_images(uniform_prob))
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def read_format_data_CelebA(
        folder: str,
        category=None,
        bs=64,
        uniform_prob=0.0):
    """Main function to read and format CelebA data

    Args:
        folder (str): path to images
        category (_type_, optional): Defaults to None.
        bs (int, optional): batch size. Defaults to 64.
        uniform_prob (float, optional): Defaults to 0.0.
    """
    is_labelled = True if category is not None else False

    # Get train dataset
    if is_labelled:
        labels = get_labels_CelebA(
            os.path.join(folder, "train/images"), category)
    else:
        labels = 'inferred'
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{folder}/train',
        labels=labels,
        label_mode='categorical' if category is not None else None,
        batch_size=bs,
        image_size=(218, 178),
        seed=7,
    )
    train_ds = _preprocess_dataset_CelebA(
        train_ds,
        augment=True,
        is_labelled=is_labelled,
        uniform_prob=uniform_prob
    )

    # Get validation dataset (no augmentation)
    if is_labelled:
        labels = get_labels_CelebA(
            os.path.join(folder, "valid/images"), category)
    else:
        labels = 'inferred'
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{folder}/valid',
        labels=labels,
        label_mode='categorical' if category is not None else None,
        batch_size=bs,
        image_size=(218, 178),
        seed=7,
    )
    valid_ds = _preprocess_dataset_CelebA(
        valid_ds, augment=False, is_labelled=is_labelled)

    return train_ds, valid_ds
