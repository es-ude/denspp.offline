import numpy as np
from glob import glob
from package.stim.imp_fitting.plot_impfit_hist import plot_boxplot_metric, plot_boxplot_params, plot_heatmap_2d_metric


def _get_params_from_saved_numpy(path2file: str) -> dict:
    """Loading the parameters from previous sweep (saved in numpy file) and transferring into common dictionary
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


def _print_results(mdata: dict) -> None:
    """Printing results into terminal"""
    unit = ('Cdl', 'Rtis', 'Zw', 'Rct')
    scale = (1e9, 1e-3, 1e-6, 1e-6)
    text_out = ('nF', 'kOhm', 'MOhm/sqrt(Hz)', 'MOhm')

    print(f"\nAnalysing data"
          f"\n==========================================")
    for idx, key in enumerate(unit):
        data_used = list()
        for data in mdata[key]:
            if len(data):
                data_used.append(data)
        data_used = np.array(data_used)

        print(f'... expected values of {scale[idx] * np.median(data_used):.3f} '
              f'(+/- {scale[idx] * np.std(data_used):.3f}) {text_out[idx]}')


def _plot_results_sweep_1d_fs(mdict: dict, name: str, used_metric: list,
                              path2save='', show_plot=True) -> None:
    """Preparing plotting with results from 1D-sweep (only frequency)"""
    fsamp = np.array(mdict['fs'], dtype=float)
    plot_boxplot_params(fsamp, 'fs', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name,
                        path2save, show_plot=show_plot)
    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_boxplot_metric(fsamp, 'fs', mdict[key], key, name, path2save, show_plot=do_plot)


def _plot_results_sweep_1d_lsb(mdict: dict, name: str, used_metric: list,
                               path2save='', show_plot=True) -> None:
    """Preparing plotting with results from 1D-sweep (only voltage of lsb)"""
    lsb = np.array(mdict['lsb'], dtype=float)
    plot_boxplot_params(lsb, 'LSB', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name,
                        path2save, show_plot=show_plot)
    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_boxplot_metric(lsb, 'LSB', mdict[key], key, name, path2save, show_plot=do_plot)


def _plot_results_sweep_2d(mdict: dict, name: str, used_metric: list, mdict_eis=(),
                           path2save='', show_plot=False) -> None:
    """Preparing plotting with results from 2D-sweep"""
    fsamp = np.array(mdict['fs'], dtype=int)
    lsb = np.array(mdict['lsb'], dtype=float)

    for key in used_metric:
        do_plot = show_plot and key == used_metric[-1]
        plot_heatmap_2d_metric(fsamp, lsb, mdict[key], key, name, path2save, mdict_eis, show_plot=do_plot)


def extract_sweep_results(path2file: str, name='results', path2save='',
                          used_metric=('MAPE', 'Rtis'), eis_params=None,
                          show_plot=False) -> None:
    """Extracting the results from sweep analysis
    (investigation on hardware resources: sampling rate and smallest voltage resolution)
    Args:
        path2file:      Path to numpy file for doing analysis
        name:           Prefix for naming the plots
        path2save:      Path for saving all plots and results
        used_metric:    List with used metrics for analysis ('MPE', 'MAPE', 'RMSRE', 'Rtis')
        eis_params:     Dictionary with parameters from the EIS measurement
        show_plot:      Showing and blocking plots
    Returns:
        None
    """
    # --- Getting data
    sweep_results = _get_params_from_saved_numpy(path2file)
    if isinstance(eis_params, dict):
        eis_params = eis_params
    else:
        eis_params = dict()

    # --- Plotting
    _print_results(sweep_results)

    num_fs = np.unique(np.array(sweep_results['fs'], dtype=int)).size
    num_lsb = np.unique(np.array(sweep_results['lsb'], dtype=float)).size
    if num_fs == 1 and num_lsb >= 1:
        _plot_results_sweep_1d_lsb(sweep_results, name, used_metric, path2save, show_plot)
    elif num_fs >= 1 and num_lsb == 1:
        _plot_results_sweep_1d_fs(sweep_results, name, used_metric, path2save, show_plot)
    elif num_fs >= 1 and num_lsb >= 1:
        _plot_results_sweep_2d(sweep_results, name, used_metric, eis_params, path2save, show_plot)
    else:
        print("Only sweeps can be done but file contains only one sample")


if __name__ == "__main__":
    choose_file = 0
    path2file = '../runs/20240906_173028_imp_fit'
    index = 'results_sweep*.npy'
    used_metrics = ('Rtis', 'Cdl', 'Rct', 'Zw', 'MAPE')

    # --- Checking for data
    used_file = glob(f'{path2file}/{index}')[choose_file]
    extract_sweep_results(used_file, name='result', path2save='runs', used_metric=used_metrics)
