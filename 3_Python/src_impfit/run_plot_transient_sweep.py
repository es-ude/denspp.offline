import numpy as np
from scipy.io.matlab import loadmat
from package.data_process.transient_resampling import quantize_transient_signal
from package.stim.imp_fitting.plot_impfit import plot_transient_stimulation


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


if __name__ == '__main__':
    fs_new = [50e3, 140e3, 1e6]
    file = '../data/tek0064ALL_MATLAB.mat'

    transient_data = read_osci_transient_from_mat(file)
    for idx, fs in enumerate(fs_new):
        show_plot = idx == len(fs_new) -1

        fs_used = fs if not fs == 0.0 else transient_data['fs']
        transient_quant = quantize_transient_signal(transient_data, fs_used, u_lsb=0.0)
        plot_transient_stimulation(fs=transient_quant['fs'], voltage=transient_quant['V'], current=transient_quant['I'],
                                   take_range=[1.25e-3, 3.5e-3], show_plot=show_plot)
