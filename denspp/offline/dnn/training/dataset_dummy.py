import numpy as np
from denspp.offline.dnn import DatasetFromFile


def generate_dummy_dataset(num_samples: int, num_window: int) -> DatasetFromFile:
    sample = np.random.randn(num_samples, num_window)
    label = np.zeros(shape=(num_samples, ))
    xpos = np.argwhere(np.mean(sample, axis=1) > 0).flatten()
    xneg = np.argwhere(np.mean(sample, axis=1) <= 0).flatten()
    label[xpos] = 1
    label[xneg] = 0
    return DatasetFromFile(
        data=sample,
        label=label,
        dict=['zero', 'one'],
        mean=np.zeros(shape=(2, num_window))
    )
