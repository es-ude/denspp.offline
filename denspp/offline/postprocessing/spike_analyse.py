import numpy as np
from denspp.offline.preprocessing import SpikeWaveform


def calc_amplitude(frames_in: SpikeWaveform) -> list:
    """Determining the min-max amplitudes of each spike frame over time
    :param frames_in:   Class SpikeWaveform with spike waveforms, positions and label
    :return:            List with [pos, ymin, ymax] for each label
    """
    fwvf = frames_in.waveform
    fpos = frames_in.xpos

    out = list()
    cluster = np.unique(frames_in.label)
    for idx, id in enumerate(cluster):
        pos = np.argwhere(frames_in.label == id).flatten()
        fsel = frames_in.waveform[pos, :]
        out.append({
            'pos': pos / frames_in.sampling_rate,
            'min': np.min(fsel, axis=0),
            'max': np.max(fsel, axis=0)
        })
    return out


def calc_spike_ticks(spike: SpikeWaveform) -> np.ndarray:
    """Determining spike ticks with cluster results
    :param spike:   Class SpikeWaveform with spike waveforms, positions and label
    :return:            Numpy array with transient ticks of available spike waveforms (sorted with label)
    """
    frames_pos = spike.xpos
    cluster_id = spike.label

    ticks = np.zeros(shape=(2, frames_pos.size), dtype=int)
    ticks[0, :] = frames_pos
    ticks[1, :] = cluster_id
    return ticks


def calc_interval_timing(ticks: np.ndarray, fs: float) -> list:
    """Calculating the interval timing (IVT) of clustered spike ticks
    :param ticks:   Numpy array with spike ticks and spike label [shape = (num_events, 2), 0: position, 1: label]
    :param fs:      Sampling rate [Hz]
    :return:        List with IVT values from ticks for each class
    """
    cluster = np.unique(ticks[1, :])
    ivt = list()
    for idx, id in enumerate(cluster):
        pos = np.argwhere(ticks[1, :] == id).flatten()
        ivt.append(np.diff(ticks[0, pos]) / fs)
    return ivt


def calc_firing_rate(ticks: np.ndarray, fs: float) -> list:
    """Calculation of the firing rate of each clustered spike ticks
    :param ticks:   Numpy array with spike ticks and spike label [shape = (num_events, 2), 0: position, 1: label]
    :param fs:      Sampling rate [Hz]
    :return:        List with firing rate values from ticks for each class
    """
    cluster = np.unique(ticks[1, :])
    out = list()
    for idx, id in enumerate(cluster):
        pos = np.argwhere(ticks[1, :] == id).flatten()
        xout = np.zeros(shape=(2, pos.size))
        xout[0, :] = ticks[0, pos] / fs
        xout[1, :] = np.concatenate((0, fs / np.diff(ticks[0, pos])), axis=None)
        out.append(xout)
    return out


def calc_autocorrelogram(ticks: np.ndarray, fs: float) -> list:
    """Calculation of the Auto-Correlogram
    :param ticks:   Numpy array with spike ticks and spike label [shape = (num_events, 2), 0: position, 1: label]
    :param fs:      Sampling rate [Hz]
    :return:        List with autocorrelated values from ticks for each class
    """
    cluster = np.unique(ticks[1, :])
    out = list()
    for idx, id in enumerate(cluster):
        isi = []
        pos = np.argwhere(ticks[1, :] == id).flatten()
        for tick_ref in pos:
            dt_isi = (ticks[0, pos] - tick_ref) / fs
            isi = np.concatenate([isi, dt_isi], axis=None)
        out.append(isi)
    return out
