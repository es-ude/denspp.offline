from os.path import join, basename
from glob import glob
from numpy import genfromtxt
import matplotlib.pyplot as plt
from statistics import mean
from denspp.offline.digital.dsp import *

def read_mdo_rawdata(file_name: str) -> dict:
    """Function for loading and handling the MDO tektronix oscilloscope data from *.csv file
    :param file_name:   Path to the waveform file
    :return:            Dictionary with waveform data
    """
    if '.csv' not in basename(file_name):
        raise FileNotFoundError(f'File {file_name} not suitable for loading regarding ending.')
    else:
        # TODO extract other info  from header?
        osci_data = genfromtxt(file_name, delimiter=',', skip_header=21)
        time = osci_data[:,0]
        current = osci_data[:,1]
        voltage = osci_data[:,2]
        fs = (len(time)-1) / (max(time)-min(time))
        osci_data = {'Time_s': osci_data[:,0], 'Current_A': osci_data[:,1], 'Voltage_V': osci_data[:,2], 'Fs': fs}
        return osci_data


def plot_mod_waveform(osci_data: dict) -> None:
    """Function for plotting oscilloscope waveforms
    :param osci_data:   Dictionary with waveform data
    :return: None
    """
    plt.plot(1e3*osci_data['Time_s'], 1e6*osci_data['Current_A'])
    plt.ylabel('Current (µA)')
    plt.xlabel('Time (ms)')
    plt.show()


def calc_charge_balance_metrics(osci_data: dict) -> tuple[float]:
    """Function for calculating metrics of charge balance phase mismatch
    :param osci_data:   Dictionary with waveform data
    :return:            Float: percentual phase mismatch, Float Injected charge imbalance in Coulomb
    """

    # TODO: Add extraction of ROI e.g. stimulus window // Removal of Amplifier Offset (determined before stim)
    snip_before_stim = osci_data['Current_A'][0:round(len(osci_data['Current_A'])/3)]
    offset = mean(snip_before_stim)

    osci_data['Current_A'] = osci_data['Current_A'] - offset
    phase_mis_perc = 100 * (max(osci_data['Current_A']) + (min(osci_data['Current_A']))) / max(osci_data['Current_A'])
    charge_imbalance = (max(osci_data['Time_s'])-min(osci_data['Time_s'])) * sum(osci_data['Current_A'])
    return phase_mis_perc, charge_imbalance


if __name__ == '__main__':
    path_to_data = glob(join(r"C:\Users\Student\sciebo - Lorenz, Nick Johannes ("
                             r"cbl162c@uni-duisburg-essen.de)@uni-duisburg-essen.sciebo.de"
                             r"\2024_DFG_RTG_InnoRetVision_B2\5_Experimente\2025_01_29_FZJ_StimPlatTest"
                             r"\20250128_HighFrqSinusoidal\Oszi", "*.csv"))
    path_file = path_to_data[6]
    print(path_file)

    mcs_data = read_mdo_rawdata(path_file)
    filterSettings = SettingsDSP(gain=1,
    fs=mcs_data['Fs'], f_filt=[20e3], n_order=2,
    type='iir', f_type='butter', b_type='lowpass',
    t_dly=0)
    DSP = DSP(filterSettings)
    DSP.use_filtfilt = True

    mcs_data['Current_A'] = DSP.filter(mcs_data['Current_A'])
    plot_mod_waveform(mcs_data)
    print(calc_charge_balance_metrics(mcs_data))