import io
from sys import getsizeof
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np


def tensor2img(img_tensor):
    if isinstance(img_tensor, tf.Tensor):
        img_tensor = img_tensor.numpy()
    img = img_tensor * 255
    img = img.astype(np.uint8)[..., 0]
    return Image.fromarray(img)


def get_size_pil(img_tensor, format='WebP'):
    img = tensor2img(img_tensor)
    buf = io.BytesIO()
    img.save(buf, format=format, lossless=True)
    return getsizeof(buf.getvalue())


def get_size_plt(img_tensor):
    buf = io.BytesIO()
    plt.clf()
    plt.imshow(img_tensor, vmin=0, vmax=1)
    plt.savefig(buf, format='png', dpi=10)
    return getsizeof(buf.getvalue())