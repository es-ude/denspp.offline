import numpy as np
from scipy.io import loadmat


def read_osci_transient_from_mat(path2file: str, i_gain:float=1e4) -> dict:
    """Loading the MATLAB files from already converted Tektronix MXO recording
    :param path2file:   String with path to file
    :param i_gain:      Floating value with transimpedance gain
    :return:            Dictionary with data of voltage 'V', current 'I', sampling_rate 'fs'
    """
    data = loadmat(path2file)
    fs_orig = 1 / np.mean(np.diff(data["DataNum"][:, 0]))
    current = data["DataNum"][:, 1] / i_gain
    voltage = data["DataNum"][:, 2]
    return {'V': voltage, 'I': current, 'fs': fs_orig}
