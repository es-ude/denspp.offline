from glob import glob
from os import mkdir
from os.path import join, exists
import numpy as np
from package.stim.imp_fitting.plot_impfit_hist import plot_boxplot_metric, plot_boxplot_params, plot_heatmap_2d_metric


def _get_params_from_mat_sweep(path2file: str) -> dict:
    """Loading the parameters from previous sweep (saved in numpy file) and transfering into common dictionary
    Args:
        path2file:  Path to numpy file with sweep results
    Returns:
        Dictionary with all results from all sweeps
    """
    # --- Preparing data processing
    keys_from_numpy = [['fs'], ['ulsb'], ['params', 'C_dl'], ['params', 'R_ct'], ['params', 'Z_war'], ['params', 'R_tis'],
                       ['metric', 'MAE'], ['metric', 'MSE'], ['metric', 'RMSE'], ['metric', 'RRMSE'],
                       ['metric', 'MAPE'], ['metric', 'MPE'], ['metric', 'RMSRE']]
    keys_to_dict = ['fs', 'lsb', 'Cdl', 'Rct', 'Zw', 'Rtis', 'MAE', 'MSE', 'RMSE', 'RRMSE', 'MAPE', 'MPE', 'RMSRE']

    # --- Generating empty dictionary
    params_dict = {}
    for key in keys_to_dict:
        params_dict.update({key: list()})

    # --- Getting the data
    for data in np.load(path2file, allow_pickle=True).flatten()[0]['results']:
        for idx, key in enumerate(params_dict.keys()):
            key_ext = keys_from_numpy[idx]
            if len(key_ext) == 2:
                params_dict[key].append(data[key_ext[0]][key_ext[1]])
            else:
                params_dict[key].append(data[key_ext[0]])

    return params_dict


def _extract_params_sweep_1d_fs(mdict: dict, name: str, path2save='',
                                used_metric=('MPE', 'MAPE', 'RMSRE'),
                                show_plot=True) -> None:
    """Preparing plotting with results from sweep"""
    # --- Output
    sel_val = -1
    unit = ('Cdl', 'Rtis', 'Zw', 'Rct')
    scale = (1e9, 1e-3, 1e-6, 1e-6)
    text_out = ('nF', 'kOhm', 'MOhm/sqrt(Hz)', 'MOhm')
    for idx, key in enumerate(unit):
        data_in = np.sort(mdict[key][sel_val], 0)
        print(f'... expected values of {scale[idx] * np.median(data_in):.3f} '
              f'(+/- {scale[idx] * np.std(data_in):.3f}) {text_out[idx]}')

    # --- Plotting
    fsamp = np.array(mdict['fs'], dtype=float)
    plot_boxplot_params(fsamp, 'fs', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name,
                        path2save, show_plot=show_plot)
    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_boxplot_metric(fsamp, 'fs', mdict[key], key, name, path2save, show_plot=do_plot)


def _extract_sweep_sweep_1d_lsb(mdict: dict, name: str, path2save='',
                                used_metric=('MPE', 'MAPE', 'RMSRE'),
                                show_plot=True) -> None:
    """Preparing plotting with results from sweep"""
    lsb = np.array(mdict['lsb'], dtype=float)
    plot_boxplot_params(lsb, 'LSB', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name,
                        path2save, show_plot=show_plot)
    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_boxplot_metric(lsb, 'LSB', mdict[key], key, name, path2save, show_plot=do_plot)


def _extract_mat_sweep_2d(mdict: dict, name: str, path2save='',
                          used_metric=('MPE', 'MAPE', 'RMSRE', 'Rtis'),
                          mdict_eis=(), show_plot=False) -> None:
    """"""
    fsamp = np.array(mdict['fs'], dtype=int)
    lsb = np.array(mdict['lsb'], dtype=float)

    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_heatmap_2d_metric(fsamp, lsb, mdict[key], key, name, path2save, mdict_eis, show_plot=do_plot)


def extract_sweep_results(path2file: str, do_plot: bool,
                          name='results', path2save='',
                          eis_params=None, used_metric=('MAPE', 'Rtis'),
                          show_plot=False) -> None:
    """Extracting the results from sweep analysis
    (investigation on hardware resources: sampling rate and smallest voltage resolution)
    Args:
        path2file:      Path to numpy file for doing analysis
        do_plot:        Boolean for generating plots
        name:           Prefix for naming the plots
        path2save:      Path for saving all plots and results
        eis_params:     Dictionary with parameters from the EIS measurement
        used_metric:    List with used metrics for analysis
        show_plot:      Showing and blocking plots
    Returns:
        None
    """
    mdict = _get_params_from_mat_sweep(path2file)
    if isinstance(eis_params, dict):
        mdict0 = eis_params
    else:
        mdict0 = dict()

    fsamp = np.array(mdict['fs'], dtype=int)
    lsb = np.array(mdict['lsb'], dtype=float)

    num_fs = np.unique(fsamp).size
    num_lsb = np.unique(lsb).size

    if num_fs == 1 and num_lsb >= 1:
        _extract_sweep_sweep_1d_lsb(mdict, name, path2save, do_plot, used_metric, show_plot=show_plot)
    elif num_fs >= 1 and num_lsb == 1:
        _extract_params_sweep_1d_fs(mdict, name, path2save, do_plot, used_metric, show_plot=show_plot)
    elif num_fs >= 1 and num_lsb >= 1:
        _extract_mat_sweep_2d(mdict, name, path2save, used_metric, mdict0, show_plot=show_plot)


if __name__ == "__main__":
    choose_file = 0
    path2file = '../runs/20240906_173028_imp_fit'
    index = 'results_sweep*.npy'
    used_metric = ('Rtis', 'Cdl', 'Rct', 'Zw', 'MAPE')

    # --- Checking for data
    folder_content = glob(f'{path2file}/{index}')
    extract_sweep_results(folder_content[choose_file], name='result', path2save='runs', used_metric=used_metric)
