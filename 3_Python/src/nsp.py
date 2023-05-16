import numpy as np

def calc_spiketicks(uin: np.ndarray, xpos: np.ndarray, cluster_id: np.ndarray) -> np.ndarray:
    """Determining spike ticks with cluster results"""
    cluster_no = np.unique(cluster_id)
    ticks = np.zeros(shape=(cluster_no.size, uin.size), dtype=int)

    idx = 0
    for val in xpos:
        ticks[cluster_id[idx], val] = 1
        idx += 1
    return ticks

def calc_interval_timing(xticks: np.ndarray, fs: float) -> list:
    """Calculating the interval timing of clustered spike ticks"""
    ivt = list()
    for idx, ticks in enumerate(xticks):
        x0 = np.where(ticks == 1)[0]
        ivt.append(np.diff(x0) / fs)
    return ivt