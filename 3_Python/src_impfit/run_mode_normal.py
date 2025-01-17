import numpy as np
from os.path import join
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat

from package.stim.imp_fitting.impfitter_handler import (ImpFitHandler, SettingsImpFit, RecommendedSettingsImpFit,
                                                        splitting_stimulation_waveforms_into_single_trials)
from package.yaml_handler import YamlConfigHandler


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


def read_impedance_from_eis(path2data: str, file_name: str) -> dict:
    """Reading the data from electrical impedance spectroscopy measurement
     Args:
         path2data:     Path to measurement file for reading
         file_name:     Name of the measurement file
    Returns:
        Dictionary with frequency 'freq' and impedance value 'Z'
    """
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
    """Merging the EIS data and calibration data
    Args:
        imp_eis:    Dictionary with impedance data from EIS measurement
        bode_cal:   Dictionary with frequency response analysis of measurement setup without DUT (shorten)
    Returns:
        Dictionary: With cleaned EIS data
    """
    imp_eis_phase_new = np.angle(imp_eis['Z']) + np.angle(bode_cal['Z'])
    imp_eis_magne_new = np.abs(imp_eis['Z'])
    imp_eis_correct = imp_eis_magne_new * np.exp(1j * imp_eis_phase_new)

    # --- Calculating the effective shunt resistance of measurement setup
    r_shunt = 500.0
    z_shunt = r_shunt * (10 ** (bode_cal['Z'] / 20.))
    imp_error = z_shunt / (z_shunt + imp_eis_correct)
    imp_new = (1. + imp_error) * imp_eis_correct
    return {'freq': imp_eis['freq'], 'Z': imp_new}


def run_over_dataset(model2fit: str, default_params: dict,
                     path2data: str, search_index: str,
                     start_sample: int=0, stop_sample: int=-1, exclude_sample=(),
                     start_folder: str='', generate_folders: bool=False,
                     ulsb_new: float=0.0, fs_new: float=0.0) -> None:
    """Run the impedancefitter for extraction of impedance data and model parameter from transient stimulation signal
    Args:
        model2fit:          Definition of the electrical model for fitting
        default_params:     Dictionary with default parameters
        path2data:          Path to all datas with transient stimulation signal
        search_index:       String with index for searching the right files
        stop_sample:        Sample position to start fitting in dataset
        stop_sample:        Sample position to stop fitting in dataset
        exclude_sample:     Sample positions to exclude from dataset
        start_folder:       Start folder for impedancefitter handler
        generate_folders:   Option if handler should generate folder
        ulsb_new:           New smallest voltage resolution (least significant bit, LSB)
        fs_new:             New sampling rate
    Returns:
        None
    """
    # --- Get a list of all data with optional limiting
    transient_files = glob(join(path2data, search_index))
    for ex_samp in exclude_sample:
        transient_files.remove(transient_files[ex_samp])
    if stop_sample == -1:
        files = transient_files[start_sample:]
    else:
        files = transient_files[start_sample:stop_sample + 1]

    # --- Init the handler
    imp_fitter = ImpFitHandler(start_folder, generate_folders)
    imp_fitter.load_fitmodel(model2fit)
    imp_fitter.define_params_default(default_params)

    # --- Processing
    ite = 0
    for file in tqdm(files):
        transient_orig = read_osci_transient_from_mat(file)
        transient_split = splitting_stimulation_waveforms_into_single_trials(transient_orig)

        imp_fitter.calculate_impedance_from_transient_dict(
            transient_signal=transient_orig, file_name=file, ratio_amp=10.0,
            u_lsb=ulsb_new, fs_new=fs_new,
            create_plot=False, show_plot=False
        )
        ite += 1


if __name__ == "__main__":
    # --- Settings
    yaml_config = YamlConfigHandler(RecommendedSettingsImpFit, yaml_name="Config_ImpFit_Normal")
    settings_impfit = yaml_config.get_class(SettingsImpFit)

    # --- Make all files
    path2data = settings_impfit.path2fits
    path2ngsolve = f'{path2data}/impedance_expected_ngsolve.csv'
    path2test0 = f'{path2data}/tek0000ALL_MATLAB_new_fit.csv'
    path2test1 = f'{path2data}/tek0000ALL_MATLAB_impedance.csv'

    # --- Step #0: Loading handler for Impedance Extraction
    imp_hndl = ImpFitHandler()
    imp_hndl.load_fitmodel(settings_impfit.model)
    imp_hndl.load_params_default(path2ngsolve, {'ct_R': 8.33e6})

    # --- Step #1: Reading the impedance data
    imp_cal0 = read_impedance_from_eis(settings_impfit.path2tran[:-12], 'Messung_INA_FRA.txt')
    imp_eis0 = read_impedance_from_eis(settings_impfit.path2tran[:-12], 'Messung_BiMEA_bm3_2E1-3E1.txt')
    imp_eis = do_eis_calibration(imp_eis0, imp_cal0)

    fit2freq = np.logspace(0, 6, 101, endpoint=True)
    imp_stm = imp_hndl.do_impedance_fit_from_predicted_csv(path2test1, fit2freq)
    imp_prd = imp_hndl.do_impedance_fit_from_params_csv(path2ngsolve, fit2freq)
    imp_hndl.plot_impedance_results(imp_eis=imp_eis, imp_stim=imp_stm, imp_mod=imp_prd,
                                    plot_name='spectrum', show_plot=True, save_plot=True)

    # --- Step #2: Processing the transient signal
    print(f'\nProcessing stimulation recordings from: {path2data}')
    print('======================================================================================================')
    run_over_dataset(settings_impfit.model, imp_hndl.get_params_default(),
                     settings_impfit.path2tran, '*_MATLAB.mat',
                     stop_sample=73, start_folder=imp_hndl.get_path2save())
