from denspp.offline.data_process.transient_resampling import do_quantize_transient
from src_impfit.imp_fitting.plot_impfit import plot_transient_stimulation
from src_impfit.data_load import read_osci_transient_from_mat


if __name__ == '__main__':
    fs_new = [50e3, 140e3, 1e6]
    file = '../data/tek0064ALL_MATLAB.mat'

    transient_data = read_osci_transient_from_mat(file)
    for idx, fs in enumerate(fs_new):
        show_plot = idx == len(fs_new) -1

        fs_used = fs if not fs == 0.0 else transient_data['fs']
        transient_quant = do_quantize_transient(transient_data, fs_used, u_lsb=0.0)
        plot_transient_stimulation(fs=transient_quant['fs'], voltage=transient_quant['V'], current=transient_quant['I'],
                                   take_range=[1.25e-3, 3.5e-3], show_plot=show_plot)
