import numpy as np
from torch import Tensor, max, min, sum, log10, sub


def calculate_snr(yin: np.ndarray, ymean: np.ndarray) -> np.ndarray:
    """Calculating the signal-to-noise ratio [dB] of the input signal compared to mean waveform
    :param yin:     Numpy array with all spike waveforms (raw data)
    :param ymean:   Numpy array with mean waveform of corresponding spike frame cluster
    :return:        Numpy array with SNR of all spike waveforms
    """
    a0 = (np.max(ymean) - np.min(ymean)) ** 2
    b0 = np.sum((yin - ymean) ** 2)
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
    return 10 * log10(a0 / b0)


def calculate_snr_tensor_waveform(input_waveform: Tensor, mean_waveform: Tensor) -> Tensor:
    """Calculation of metric Signal-to-Noise ratio (SNR) of defined input and reference waveform
    Args:
        input_waveform:     Tensor array with input waveform
        mean_waveform:      Tensor array with real mean waveform from dataset
    Return:
        Tensor with differential Signal-to-Noise ratio (SNR) of applied waveforms
    """
    return calculate_snr_tensor(input_waveform, mean_waveform)


def calculate_dsnr_tensor_waveform(input_waveform: Tensor, pred_waveform: Tensor, mean_waveform: Tensor) -> Tensor:
    """Calculation of metric different Signal-to-Noise ratio (SNR) between defined input and predicted to reference waveform
    Args:
        input_waveform:     Tensor array with input waveform
        pred_waveform:      Tensor array with predicted waveform from model
        mean_waveform:      Tensor array with real mean waveform from dataset
    Return:
        Tensor with differential Signal-to-Noise ratio (SNR) of applied waveforms
    """
    snr_in = calculate_snr_tensor(input_waveform, mean_waveform)
    snr_out = calculate_snr_tensor(pred_waveform, mean_waveform)
    return sub(snr_out, snr_in)
