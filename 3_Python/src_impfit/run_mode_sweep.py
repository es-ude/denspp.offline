import numpy as np
from glob import glob
from os.path import join

from src_impfit.run_mode_normal import run_over_dataset, read_impedance_from_eis, do_eis_calibration
from package.metric import (calculate_error_mae, calculate_error_mse, calculate_error_rmse, calculate_error_rrmse,
                            calculate_error_mape, calculate_error_mpe, calculate_error_rmsre)
from package.stim.imp_fitting.impfitter_handler import ImpFit_Handler, load_params_from_csv
from package.stim.imp_fitting.plot_impfit_hist import plot_hist_impedance


def do_single_run(fit_model: str, params_default: dict,
                  path2run: str, path2data: str, index_search: str,
                  imp_eis: dict, fs_used: float, ulsb_used: float,
                  take_samples=None) -> dict:
    """"""
    if take_samples is None:
        take_samples = [0, -1]

    # --- Part 0: Generate folder for sampling rate
    path2save = join(path2run, f'results_{int(1e-3 * fs_used)}kHz_{int(1e6 * ulsb_used)}uV')

    # --- Part 1: Getting the parameters with MSE determination
    run_over_dataset(fit_model, params_default,
                     path2data, index_search,
                     fs_new=fs_used, ulsb_new=ulsb_used, start_folder=path2save, generate_folders=True,
                     start_sample=take_samples[0], stop_sample=take_samples[1])

    # --- Part 2: Do statistic run
    path2params = glob(join(path2save, "model_param", "*_params.csv"))
    param = extract_csv_param(path2params, path2save)
    metric = extract_csv_metric(path2params, imp_eis)

    # --- Part 3: Calculate error, compared to measurement
    return {'fs': fs_used, 'ulsb': ulsb_used, 'params': param, 'metric': metric}


def run_sweep(fit_model: str, default_params: dict,
              path2run: str, path2data: str, index_search: str, imp_eis: dict,
              sweep_fs: np.ndarray | list, sweep_ulsb: np.ndarray | list,
              take_samples=None) -> None:
    """"""
    # --- Generating the sweep parameter
    if isinstance(sweep_fs, np.ndarray):
        sweep_fs = sweep_fs.tolist()
    if isinstance(sweep_ulsb, np.ndarray):
        sweep_ulsb = sweep_ulsb.tolist()

    sweep_params = list()
    for fs_now in sweep_fs:
        for ulsb_now in sweep_ulsb:
            sweep_params.append({'fs': fs_now, 'lsb': ulsb_now})

    # --- Running Code
    results = list()
    # --- Processing the data
    print(f'\nProcessing stimulation recordings from: {path2data}')
    print('======================================================================================================')
    for ite, params in enumerate(sweep_params):
        print(f'... run #{ite} ({ite / len(sweep_params) * 100:.2f} %) @ fs = {1e-3 * params["fs"]:.1f} kHz and '
              f'u_lsb = {1e6 * params["lsb"]:.1f} µV')
        results.append(do_single_run(
            fit_model, default_params,
            path2run, path2data, index_search, imp_eis,
            params['fs'], params['lsb'], take_samples
        ))

    # --- Saving
    file_name = 'results_sweep'
    if len(sweep_ulsb) > 1 and len(sweep_ulsb) > 1:
        file_name += '_2d_frq_lsb'
    elif len(sweep_ulsb) == 1 and len(sweep_ulsb) > 1:
        file_name += '_1d_frq'
    elif len(sweep_ulsb) > 1 and len(sweep_ulsb) == 1:
        file_name += '_1d_lsb'
    else:
        file_name += '_0d'
    np.save(join(path2run, f'{file_name}.npy'), {'results': results}, allow_pickle=True)


def extract_csv_metric(files: list, imp_mea: dict) -> dict:
    """Extracting the metric values from csv impfitter runs"""
    # Generating new dict with metrics (first functional call, second empty results)
    metric_used = {
        'MAE': calculate_error_mae, 'MSE': calculate_error_mse,
        'RMSE': calculate_error_rmse, 'RRMSE': calculate_error_rrmse,
        'MAPE': calculate_error_mape, 'MPE': calculate_error_mpe,
        'RMSRE': calculate_error_rmsre
    }
    metric_dict = {}
    for key in metric_used.keys():
        metric_dict.update({key: list()})

    # Extracting metrics from data
    for idx, file in enumerate(files):
        imp_fit = imp_hndl.do_impedance_fit_from_params_csv(file, imp_mea['freq'])
        imp_fit_used = np.abs(imp_fit['Z'])
        imp_mea_used = np.abs(imp_mea['Z'])
        for key in metric_used:
            metric_dict[key].append(metric_used[key](imp_mea_used, imp_fit_used))

    return metric_dict


def extract_csv_param(files_fit: list, path2save='', do_plot=False, do_print=False) -> dict:
    """Extracting the electrical target values from csv impfitter runs"""
    # --- Reading parameters
    c_dl = np.zeros((len(files_fit),), dtype=float)
    r_ct = np.zeros((len(files_fit),), dtype=float)
    z_war = np.zeros((len(files_fit),), dtype=float)
    r_tis = np.zeros((len(files_fit),), dtype=float)

    for idx, file in enumerate(files_fit):
        params = load_params_from_csv(file)
        c_dl[idx] = params['dl_C']
        r_ct[idx] = params['ct_R']
        z_war[idx] = params['war_Aw']
        r_tis[idx] = params['tis_R']

    if do_plot:
        plot_hist_impedance(r_tis, z_war, c_dl, r_ct, '', path2save)

    # --- Printing
    if do_print:
        print("\nResults of Parameter Extraction:"
              "\n=============================================================")
        print(f"C_dl = {1e9 * np.mean(c_dl):.4f} (+/- {1e9 * np.std(c_dl):.3f}) nF")
        print(f"R_ct = {1e-6 * np.mean(r_ct):.4f} (+/- {1e-6 * np.std(r_ct):.3f}) MOhm")
        print(f"Z_war = {1e-6 * np.mean(z_war):.4f} (+/- {1e-6 * np.std(z_war):.3f}) MOhm/sqrt(Hz)")
        print(f"R_tis = {1e-3 * np.mean(r_tis):.4f} (+/- {1e-3 * np.std(r_tis):.3f}) kOhm\n")
    return {'C_dl': c_dl, 'R_ct': r_ct, 'Z_war': z_war, 'R_tis': r_tis}


if __name__ == '__main__':
    # --- Settings
    do_plot = False
    take_samples = [0, 4]
    fs_resample = np.logspace(5, 7.4, num=2, endpoint=True, dtype=np.float32)
    ulsb_resample = np.logspace(-5, -1.6, num=2, endpoint=True, dtype=np.float32)

    set_ifitter = "R_tis + W_war + parallel(R_ct, C_dl)"

    path2data = 'C:/HomeOffice/Austausch_Rostock/TransienteMessungen/180522_Messung/1_Messdaten'
    index_search = '*_MATLAB.mat'
    path2ngsolve = f'../../2_Data/00_ImpedanceFitter/impedance_expected_ngsolve.csv'
    path2eis_params = f'../../2_Data/00_ImpedanceFitter/Messung_BiMEA_bm3_2E1-3E1_params.csv'

    # --- Step #0: Loading EIS parameter (extracted)
    eis_params = load_params_from_csv(path2eis_params)

    # --- Step #1: Extraction Impedance from real measurement
    imp_hndl = ImpFit_Handler()
    imp_hndl.load_fitmodel(set_ifitter)
    imp_hndl.load_params_default(path2ngsolve, {'ct_R': 8.33e6})

    # Reading the impedance data from EIS
    imp_cal0 = read_impedance_from_eis(path2data[:-12], 'MESSUNG_INA_FRA.txt')
    imp_eis0 = read_impedance_from_eis(path2data[:-12], 'Messung_BiMEA_bm3_2E1-3E1.txt')
    imp_eis = do_eis_calibration(imp_eis0, imp_cal0)
    path2save = imp_hndl.get_path2save()

    # Reading the impedance from other sources
    fit2freq = np.logspace(0, 6, 101, endpoint=True)
    imp_prd = imp_hndl.do_impedance_fit_from_params(imp_hndl.get_params_default(), fit2freq)
    imp_fit = imp_hndl.do_impedance_fit_from_params_csv(path2eis_params, fit2freq)
    imp_hndl.plot_impedance_results(imp_eis=imp_eis, imp_fit=imp_fit, imp_mod=imp_prd,
                                    plot_name='spectrum', show_plot=True, save_plot=True)

    # --- Step #2: Doing step with integrated analysis
    run_sweep(set_ifitter, imp_hndl.get_params_default(),
              path2save, path2data, index_search, imp_eis,
              fs_resample, ulsb_resample, take_samples
              )
    print('This is the End')
