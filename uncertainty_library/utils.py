import tensorflow as tf
import numpy as np


def remove_test_time_dropout(model_drop):
    config = model_drop.get_config()
    for layer in config['layers']:
        if 'Dropout' in layer['class_name']:
            layer['inbound_nodes'][0][0][3]['training'] = False
    model = tf.keras.Model().from_config(config)
    model.set_weights(model_drop.get_weights())
    return model


def ds2numpy(ds, max_num_batches):
    x, y = [], []
    for batch in ds.take(max_num_batches):
        x.append(batch[0].numpy())
        y.append(batch[1].numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y
