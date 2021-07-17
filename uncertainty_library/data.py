import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


# Data reading and formatting
# =============================================================================

def read_format_data(dataset: str):
    if dataset == 'MNIST':
        ds = tf.keras.datasets.mnist
    elif dataset == 'FashionMNIST':
        ds = tf.keras.datasets.fashion_mnist
    else:
        raise ValueError("`dataset` should be either `MNIST` or `FashionMNIST`, "
                         f"but {dataset} ws provided")

    # Load and scale data
    (x_train, y_train), (x_test, y_test) = ds.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)

    # Create tensors
    x_train = tf.convert_to_tensor(x_train, dtype='float32')
    x_test = tf.convert_to_tensor(x_test, dtype='float32')
    y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


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


def get_labels_CelebA(folder, category):
    with open('list_attr_celeba.txt', 'r') as attr_file:
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
        attributes_dict.update({file_name: [int(attr == '1') for attr in attributes]})

    file_list = sorted(os.listdir(folder))
    category_idx = attributes_names.index(category)
    labels = []
    for file_name in file_list:
        label = attributes[file_name][category_idx]
        label_onehot = [0, 0]
        label_onehot[label] = 1
        labels.append(label_onehot)

    return labels


def _preprocess_dataset(ds, augment=True, is_labelled=True):
    # wrapper for labelled datasets
    if is_labelled:
        def decorator(func):
            def wrapped(image, label):
                return func(image), label
            return wrapped
    else:
        def decorator(func):
            return func

    # Prepare data augmentation layer
    augmentation = tf.keras.models.Sequential(
        [
            preprocessing.RandomRotation(factor=0.05),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
            preprocessing.RandomFlip(mode='horizontal'),
        ],
        name='augmentation'
    )

    # First slicing crops the image to the centre of the face which is important
    # for random rotation in the augmentation layer. The second slicing crops
    # the image further to size 128x128 which removes any padding artefacts.
    ds = ds.map(decorator(lambda image: image[:, 27: 205, :, :]))
    if augment:
        ds = ds.map(decorator(augmentation), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(decorator(lambda image: image[:, 25: -25, 25: -25, :]))
    ds = ds.map(decorator(lambda image: tf.clip_by_value(image, 0.0, 255.0) / 255.0))
    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def read_format_data_CelebA(folder, category=None, bs=64):
    is_labelled = True if category is not None else False

    # Get train dataset
    if is_labelled:
        labels = get_labels_CelebA(os.path.join(folder, "train/images"), category)
    else:
        labels = 'inferred'
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{folder}/train',
        labels=labels,
        label_mode='categorical' if category is not None else None,
        batch_size=bs,
        image_size=(218, 178),
    )
    train_ds = _preprocess_dataset(train_ds, augment=True, is_labelled=is_labelled)

    # Get validation dataset (no augmentation)
    if is_labelled:
        labels = get_labels_CelebA(os.path.join(folder, "valid/images"), category)
    else:
        labels = 'inferred'
    valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
        f'{folder}/valid',
        labels=labels,
        label_mode='categorical' if category is not None else None,
        batch_size=bs,
        image_size=(218, 178),
        shuffle=False,
    )
    valid_ds = _preprocess_dataset(valid_ds, augment=False, is_labelled=is_labelled)

    return train_ds, valid_ds
