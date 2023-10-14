import numpy as np


# TODO: Weitere Methoden (wie Kreuz- und Autokorrelation einfügen)
def calc_spiketicks(uin: np.ndarray, xpos: np.ndarray, cluster_id: np.ndarray, do_short=False) -> np.ndarray:
    """Determining spike ticks with cluster results"""
    cluster_no = np.unique(cluster_id)

    if do_short:
        ticks = np.zeros(shape=(xpos.size, 2), dtype=np.int8)
        ticks[:, 0] = xpos
        ticks[:, 1] = cluster_id
    else:
        # --- Performing the long type
        ticks = np.zeros(shape=(cluster_no.size, uin.size), dtype=np.int8)
        for idx, val in enumerate(xpos):
            ticks[cluster_id[idx], val] = 1

    return ticks


def calc_amplitude(frame: np.ndarray, xpos: np.ndarray) -> list:
    """Determining the min-max amplitudes of each spike frame over time"""
    amp = list()
    for idx, frame0 in enumerate(frame):
        amp.append([xpos[idx], np.min(frame0), np.max(frame0)])
    
    return amp


def calc_interval_timing(xticks: np.ndarray, fs: float) -> list:
    """Calculating the interval timing of clustered spike ticks"""
    ivt = list()
    for idx, ticks in enumerate(xticks):
        x0 = np.where(ticks == 1)[0]
        ivt.append(np.diff(x0) / fs)
    return ivt


def calc_firing_rate(xticks: np.ndarray, fs: float) -> list:
    """Calculation of the firing rate of each clustered spike ticks"""
    out = list()
    for idx, ticks in enumerate(xticks):
        x0 = np.where(ticks == 1)[0]
        xout = np.zeros(shape=(2, x0.size))
        xout[0, :] = x0 / fs
        xout[1, :] = np.concatenate((0, fs / np.diff(x0)), axis=None)
        out.append(xout)
    return out


def calc_autocorrelogram(xticks: np.ndarray, fs: float) -> list:
    """Calculation of the Auto-Correlogram"""
    out = list()
    for idx, ticks0 in enumerate(xticks):
        isi = []
        ticks = np.where(ticks0 == 1)[0]
        for tick_ref in ticks:
            dt_isi = ticks - tick_ref
            isi = np.concatenate([isi, dt_isi], axis=None)
        out.append(isi / fs)
    return out


def calc_crosscorrelogram(xticks: np.ndarray, xref: np.ndarray, fs: float) -> list:
    """Calculation of the Cross-Correlogram"""
    out = list()
    tick_ref0 = np.where(xref == 1)[0]
    for idx, ticks0 in enumerate(xticks):
        isi = []
        ticks = np.where(ticks0 == 1)[0]
        for tick_ref in tick_ref0:
            dt_isi = ticks - tick_ref
            isi = np.concatenate([isi, dt_isi], axis=None)
        out.append(isi / fs)
    return out
