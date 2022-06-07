import numpy as np


def ds2numpy(ds, max_num_batches):
    x, y = [], []
    for batch in ds.take(max_num_batches):
        x.append(batch[0].numpy())
        y.append(batch[1].numpy())
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    return x, y
