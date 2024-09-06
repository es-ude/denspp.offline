import numpy as np
from os.path import join
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat

from package.stim.imp_fitting.impfitter_handler import ImpFit_Handler


def _read_osci_transient_from_mat(path2file: str) -> dict:
    """Loading the MATLAB files from already converted Tektronix MXO recording"""
    data = loadmat(path2file)
    fs_orig = 1 / np.mean(np.diff(data["DataNum"][:, 0]))
    current = 1e-4 * data["DataNum"][:, 1]
    voltage = data["DataNum"][:, 2]
    return {'V': voltage, 'I': current, 'fs': fs_orig}


def read_impedance_from_eis(path2data: str, file_name: str) -> dict:
    """Reading the impedance data"""
    # --- Reading data
    file = open(join(path2data, file_name), 'r')
    content = file.readlines()
    file.close()

    # --- Processing data
    f0 = []
    imp_abs = []
    imp_pha = []
    for line in content:
        if not line[0] == '>':
            data = line[:-1].split(',')
            f0.append(float(data[0]))
            imp_abs.append(float(data[3]))
            imp_pha.append(float(data[4]))

    abs0 = np.array(imp_abs)
    phase0 = np.array(imp_pha)

    f0 = np.array(f0)
    imp0 = abs0 * np.exp(1j * phase0 / 180 * np.pi)
    return {'freq': f0, 'Z': imp0}


def do_eis_calibration(imp_eis: dict, bode_cal: dict) -> dict:
    """
    Info: Calibration data is used for calibrating the measurement setup without DUT (shorten)
    """
    imp_eis_phase_new = np.angle(imp_eis['Z']) + np.angle(bode_cal['Z'])
    imp_eis_magne_new = np.abs(imp_eis['Z'])
    imp_eis_correct = imp_eis_magne_new * np.exp(1j * imp_eis_phase_new)

    # --- Calculating the effective shunt resistance of measurement setup
    r_shunt = 100.0
    r_input = 50.0
    z_shunt = r_shunt * (10 ** (bode_cal['Z'] / 20.))
    imp_error = z_shunt / (z_shunt + imp_eis_correct + r_input)
    imp_new = (1. + imp_error) * imp_eis_correct
    return {'freq': imp_eis['freq'], 'Z': imp_new}


def run_over_dataset(model2fit: str, path2data: str, search_index: str,
                     start_sample=0, stop_sample=-1, exclude_sample=(),
                     path2ngmodel='', start_folder='', generate_folders=False,
                     u_lsb=0.0, fs_new=0.0) -> None:
    """"""
    # --- Get a list of all data with optional limiting
    transient_files = glob(join(path2data, search_index))
    for ex_samp in exclude_sample:
        transient_files.remove(transient_files[ex_samp])
    if stop_sample == -1:
        files = transient_files[start_sample:]
    else:
        files = transient_files[start_sample:stop_sample + 1]

    # --- Init the handler
    imp_fitter = ImpFit_Handler(start_folder, generate_folders)
    imp_fitter.load_fitmodel(model2fit)
    imp_fitter.define_params_default({'ct_R': 0.0, 'tis_R': 0.0, 'dl_C': 0.0, 'war_Aw': 0.0})

    # --- Processing the data
    print(f'\nProcessing stimulation recordings from: {path2data}')
    print('======================================================================================================')
    ite = 0
    for file in tqdm(files):
        transient_orig = _read_osci_transient_from_mat(file)
        imp_fitter.calculate_impedance_from_transient_dict(
            transient_signal=transient_orig, file_name=file, ratio_amp=10.0,
            u_lsb=u_lsb, fs_new=fs_new,
            create_plot=False, show_plot=False
        )
        ite += 1


if __name__ == "__main__":
    set_ifitter = "R_tis + W_war + parallel(R_ct, C_dl)"

    path2data = '../../2_Data/00_ImpedanceFitter'
    path2ngsolve = f'{path2data}/impedance_expected_ngsolve.csv'
    path2test0 = f'{path2data}/tek0000ALL_MATLAB_new_fit.csv'
    path2test1 = f'{path2data}/tek0000ALL_MATLAB_impedance.csv'

    path2data_all = 'C:/HomeOffice/Austausch_Rostock/TransienteMessungen/180522_Messung/1_Messdaten'

    # --- Loading handler for Impedance Extraction
    imp_hndl = ImpFit_Handler()
    imp_hndl.load_fitmodel(set_ifitter)
    imp_hndl.load_params_default(path2ngsolve, {'ct_R': 8.33e6})

    # --- Reading the impedance data
    imp_cal0 = read_impedance_from_eis(path2data_all[:-12], 'Messung_INA_FRA.txt')
    imp_eis0 = read_impedance_from_eis(path2data_all[:-12], 'Messung_BiMEA_bm3_2E1-3E1.txt')
    imp_eis = do_eis_calibration(imp_eis0, imp_cal0)

    fit2freq = np.logspace(0, 6, 101, endpoint=True)
    imp_stm = imp_hndl.do_impedance_fit_from_predicted(path2test1, fit2freq)
    imp_prd = imp_hndl.do_impedance_fit_from_params(path2ngsolve, fit2freq)
    imp_hndl.plot_impedance_results(imp_eis=imp_eis, imp_stim=imp_stm, imp_mod=imp_prd,
                                    plot_name='spectrum', show_plot=True, save_plot=True)

    # --- Processing the transient signal
    run_over_dataset(set_ifitter, path2data_all, '*_MATLAB.mat',
                     stop_sample=73, path2ngmodel=path2ngsolve, start_folder=imp_hndl.get_path2save())
