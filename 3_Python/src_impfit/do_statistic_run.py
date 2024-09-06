from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np

from package.stim.imp_fitting.plot_impfit_hist import *


def _get_params_from_mat_sweep(path2file: str) -> dict:
    """"""
    data0 = np.load(path2file, allow_pickle=True).flatten()[0]['results']

    fsamp = list()
    ulsb = list()
    c_dl0 = list()
    r_ct0 = list()
    z_wa0 = list()
    r_ti0 = list()
    mae = list()
    mse = list()
    rmse = list()
    rrmse = list()
    mape = list()
    mpe = list()
    rmsre = list()

    # Getting the data
    for idx, data in enumerate(data0):
        fsamp.append(data['fs'])
        ulsb.append(data['ulsb'])
        if data['params']['C_dl'].size != 0:
            c_dl0.append(data['params']['C_dl'])
            r_ct0.append(data['params']['R_ct'])
            z_wa0.append(data['params']['Z_war'])
            r_ti0.append(data['params']['R_tis'])
            mae.append(data['metric']['MAE'])
            mse.append(data['metric']['MSE'])
            rmse.append(data['metric']['RMSE'])
            rrmse.append(data['metric']['RRMSE'])
            mape.append(data['metric']['MAPE'])
            mpe.append(data['metric']['MPE'])
            rmsre.append(data['metric']['RMSRE'])
        else:
            c_dl0.append(np.ndarray(0))
            r_ct0.append(np.ndarray(0))
            z_wa0.append(np.ndarray(0))
            r_ti0.append(np.ndarray(0))
            mae.append(np.ndarray(0))
            mse.append(np.ndarray(0))
            rmse.append(np.ndarray(0))
            rrmse.append(np.ndarray(0))
            mape.append(np.ndarray(0))
            mpe.append(np.ndarray(0))
            rmsre.append(np.ndarray(0))
    del data0, data, idx

    return {'fs': fsamp, 'lsb': ulsb,
            'Cdl': c_dl0, 'Rct': r_ct0, 'Zw': z_wa0, 'Rtis': r_ti0,
            'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'RRMSE': rrmse, 'MAPE': mape, 'MPE': mpe, 'RMSRE': rmsre}


def _extract_mat_sweep_1d_fs(mdict: dict, name: str, path2save='', do_plot=False,
                             used_metric=('MPE', 'MAPE', 'RMSRE'),
                             mdict_eis=()) -> None:
    """Extracting """
    fsamp = np.array(mdict['fs'], dtype=int)
    if do_plot:
        # --- Generate new folder
        path2save_new = join(path2save, 'results_sweep1d_frq')
        if not exists(path2save_new):
            mkdir(path2save_new)
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
        plot_boxplot_params(fsamp, 'fs', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name, path2save)
        for key in used_metric:
            plot_boxplot_metric(fsamp, 'fs', mdict[key], key, name, path2save_new)


def _extract_mat_sweep_1d_lsb(mdict: dict, name: str, path2save='', do_plot=False,
                              used_metric=('MPE', 'MAPE', 'RMSRE')) -> None:
    """"""
    lsb = np.array(mdict['lsb'], dtype=float)
    if do_plot:
        path2save_new = join(path2save, 'results_sweep1d_lsb')
        if not exists(path2save_new):
            mkdir(path2save_new)
        plot_boxplot_params(lsb, 'LSB', mdict['Rtis'], mdict['Zw'], mdict['Cdl'], mdict['Rct'], name, path2save_new)
        for key in used_metric:
            plot_boxplot_metric(lsb, 'LSB', mdict[key], key, name, path2save_new)


def _extract_mat_sweep_2d(mdict: dict, name: str, path2save='',
                          used_metric=('MPE', 'MAPE', 'RMSRE', 'Rtis'),
                          mdict_eis=()) -> None:
    """"""
    fsamp = np.array(mdict['fs'], dtype=int)
    lsb = np.array(mdict['lsb'], dtype=float)

    for key in used_metric:
        plot_heatmap_2d_metric(fsamp, lsb, mdict[key], key, name, path2save, mdict_eis, show_plot=key == used_metric[-1])


def extract_mat_sweep(path2file: str, name: str, path2save='',
                      do_plot=False, eis_params=None,
                      used_metric=('MAPE', 'Rtis')) -> None:
    """"""
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
        _extract_mat_sweep_1d_lsb(mdict, name, path2save, do_plot, used_metric)
    elif num_fs >= 1 and num_lsb == 1:
        _extract_mat_sweep_1d_fs(mdict, name, path2save, do_plot, used_metric)
    elif num_fs >= 1 and num_lsb >= 1:
        _extract_mat_sweep_2d(mdict, name, path2save, used_metric, mdict0)


if __name__ == "__main__":
    choose_file = 0
    path2file = '../runs/20240906_173028_imp_fit'
    index = 'results_sweep*.npy'
    used_metric = ('Rtis', 'Cdl', 'Rct', 'Zw', 'MAPE')
    use_params_eis = True

    # --- Checking for data
    folder_content = glob(f'{path2file}/{index}')
    extract_mat_sweep(folder_content[choose_file], 'result', 'runs', do_plot=True,
                      eis_params=use_params_eis, used_metric=used_metric)
