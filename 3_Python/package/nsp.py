import numpy as np


# TODO: Weitere Methoden (wie Kreuz- und Autokorrelation einfügen)
def calc_spiketicks(frames_in: list, do_short=True, out_transient_size=0) -> np.ndarray:
    """Determining spike ticks with cluster results"""
    frames_pos = frames_in[1]
    cluster_id = frames_in[2]
    cluster_no = np.unique(cluster_id)

    # --- Generation of spike ticks
    if do_short:
        ticks = np.zeros(shape=(frames_pos.size, 2), dtype=int)
        ticks[:, 0] = frames_pos
        ticks[:, 1] = cluster_id
    else:
        # --- Performing the long type
        ticks = np.zeros(shape=(cluster_no.size, out_transient_size), dtype=np.int8)
        for idx, val in enumerate(frames_pos):
            ticks[cluster_id[idx], val] = 1

    return ticks


def calc_amplitude(frames_in: list) -> list:
    """Determining the min-max amplitudes of each spike frame over time"""
    frames0_in = frames_in[0]
    frames_pos = frames_in[1]
    cluster_id = frames_in[2]
    amp = list()
    cluster_no = np.unique(cluster_id)
    for id in cluster_no:
        selx = np.where(cluster_id == id)[0]
        amp_id = list()
        sel_frames = frames0_in[selx, :]
        for idx, frame0 in enumerate(sel_frames):
            amp_id.append([frames_pos[selx[idx]], np.min(frame0), np.max(frame0)])
        amp.append(amp_id)

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
