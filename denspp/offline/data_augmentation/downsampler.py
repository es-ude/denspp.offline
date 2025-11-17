import numpy as np


def augmentation_downsampling(
        data: np.ndarray,
        label: np.ndarray,
        n_downsampling: int,
        drop_samples: bool=False
) -> tuple[np.ndarray, np.ndarray]:
    """Function for data augmentation using downsampling.
    :param data:            Numpy array with data from dataset with shape [num_samples, num_ch (optional), num_timepoints]
    :param label:           Numpy array with labels from dataset with shape [num_samples, ]
    :param n_downsampling:  Integer with level for downsampling (0: error, 1: no change, 2: reduced by 2, 3: reduced by 3, ...)
    :param drop_samples:    Boolean, whether to drop samples (true) or augment samples (false)
    :return:                Tuple with data and label for the new augmented dataset using downsampling.
    """
    if n_downsampling < 1:
        raise ValueError('n_downsampling must be >= 1')
    elif n_downsampling == 1:
        return data, label
    else:
        data_used = data[..., 0::n_downsampling]
        label_used = label

        if not drop_samples:
            data_new = data[..., 1::n_downsampling]
            label_new = label
            for idx in range(2, n_downsampling):
                data_lost = data[..., idx::n_downsampling]
                if data_new.shape[-1] > data_lost.shape[-1]:
                    append_size = data_new.shape[-1] - data_lost.shape[-1]
                    data_padd = np.zeros(data_new.shape[:-1] + (append_size, ))
                    data_lost = np.concatenate([data_lost, data_padd], axis=-1)

                data_new = np.concatenate([data_new, data_lost], axis=0)
                label_new = np.concatenate([label_new, label], axis=0)
            data_out = np.concatenate([data_used, data_new], axis=0)
            label_out = np.concatenate([label_used, label_new], axis=0)
            return data_out, label_out
        else:
            return data_used, label_used
