import numpy as np
from torch import Tensor, max, min, sum, log10, sub, div


def calculate_snr(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform
    :param data:    Numpy array with all spike waveforms (raw data)
    :param mean:    Numpy array with mean waveform of corresponding spike frame cluster
    :return:        Numpy array with SNR of all spike waveforms
    """
    a0 = (np.max(mean) - np.min(mean)) ** 2
    b0 = np.sum((data - mean) ** 2)
    return 10 * np.log10(a0 / b0)


def calculate_snr_tensor(data: Tensor, mean: Tensor) -> Tensor:
    """Calculating the Signal-to-Noise (SNR) ratio of the input data
    Args:
        data:   Tensor with raw data / frame
        mean:   Tensor with class-specific mean data / frame
    Return:
        Tensor with SNR value
    """
    max_values, _ = max(mean, dim=1)
    min_values, _ = min(mean, dim=1)
    a0 = (max_values - min_values) ** 2
    b0 = sum((data - mean) ** 2, dim=1)
    return 10 * log10(div(a0, b0))


def calculate_dsnr_tensor(data: Tensor, pred: Tensor, mean: Tensor) -> Tensor:
    """Calculation of metric different Signal-to-Noise ratio (SNR) between defined input and predicted to reference waveform
    Args:
        data:   Tensor array with input waveform
        pred:   Tensor array with predicted waveform from model
        mean:   Tensor array with real mean waveform from dataset
    Return:
        Tensor with differential Signal-to-Noise ratio (SNR) of applied waveforms
    """
    snr_in = calculate_snr_tensor(data, mean)
    snr_out = calculate_snr_tensor(pred, mean)
    return sub(snr_out, snr_in)


def calculate_snr_cluster(frames_in: np.ndarray, frames_cl: np.ndarray, frames_mean: np.ndarray) -> np.ndarray:
    """Calculating the cluster-specific Signal-to-Noise Ratio (SNR) for all frames
    :param frames_in:   Numpy array with spike frames
    :param frames_cl:   Numpy array with cluster label to each spike frame
    :param frames_mean: Numpy array with mean waveforms of cluster
    :return:            Numpy array with SNR value for each sample for {min, mean, max}
    """
    id_cluster, num_cluster = np.unique(frames_cl, return_counts=True)

    cluster_snr = np.zeros(shape=(num_cluster.size, 4), dtype=float)
    for idx, id in enumerate(id_cluster):
        indices = np.where(frames_cl == id)[0]
        snr0 = np.zeros(shape=(indices.size,), dtype=float)
        for i, frame in enumerate(frames_in[indices, :]):
            snr0[i] = calculate_snr(frame, frames_mean[id, :])

        cluster_snr[idx, 0] = np.min(snr0)
        cluster_snr[idx, 1] = np.mean(snr0)
        cluster_snr[idx, 2] = np.max(snr0)
    return cluster_snr
