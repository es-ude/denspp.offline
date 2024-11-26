import os
import numpy as np
import h5py
from glob import glob
from matplotlib import pyplot as plt
from datetime import datetime
from tqdm import tqdm
from package.data_process.frame_normalization import DataNormalization


def load_fzj_onoff_waveforms(path2folder: str, path2data: str='', quantize_bitwidth: int=16) -> None:
    """Processing the spike frames from the wildtype experiment with Research Center Juelich
    Args:
        path2folder:        Path to folder which contains the originial data
        path2data:          Path for saving the output dictionary
        quantize_bitwidth:  Bitwidth of quantization after min/max normalisation
    Return:
        Dictionary with quantized spike frames, sampling rate and amplitudes
    """
    file_index = 'waveforms_*.mat'
    file_list = glob(os.path.join(path2folder, file_index))

    if len(file_list) == 0:
        raise RuntimeError("Path or data does not exist. Please check the path!")

    data_onoff = {'date_created': datetime.now().strftime("%Y%m%d_%H%M%S"), 'sampling_rate_khz': 25}
    data_onoff.update({'electrode': 'mcs60mea'})

    data_temp = dict()
    for file in tqdm(file_list):
        # --- Loading
        hndlr = h5py.File(file, 'r')
        data0 = np.array(hndlr.get('tot_spike_matrix'))
        data1 = np.transpose(data0)
        del data0

        # --- Processing for DataSize reduction
        norm_hndl = DataNormalization("minmax", "bipolar")
        pre_scaler = 2 ** (quantize_bitwidth-1) - 1
        data_dig = np.array(pre_scaler * norm_hndl.normalize(data1), dtype=np.int16)
        amp_array = norm_hndl.get_peak_amplitude_values()

        hndlr.close()
        del data1, hndlr, norm_hndl

        label = file.split('.')[0].split("_")[-1]
        data_temp.update({label: {'raw': data_dig, 'amp_peak_uV': amp_array}})

    data_right_format = get_data_and_label_from_rawdata(data_temp)
    data_onoff.update(data_right_format)
    np.save(os.path.join(path2data, 'fzj_onoff_waveforms.npy'), data_onoff, allow_pickle=True)


def get_data_and_label_from_rawdata(data_raw: dict) -> dict:
    """Function for bringing spike frames and label from rawdata (read from MCS FZJ data structure)
    Args:
        data_raw:   Dictionary with raw data, should contain key 'amp_peak_uV' (Frame Peaks), 'raw' (all spike frames)
    Return:
        Dictionary for generating dataset for DNN training with keys 'peak' (spike frame amplitude), 'raw' (spike frames) and 'label' (available labels)
    """
    data_keys = ['OFFsus', 'OFFtra', 'ONsus', 'ONtra', 'ONOFF']

    dict_dataset = {'raw': np.zeros((1, 1)), 'label': np.zeros((1, 1)), 'peak': np.zeros((1, 1)), 'class': dict()}
    for idx, key in enumerate(data_keys):
        used_peak = data_raw[key]['amp_peak_uV']
        used_data = data_raw[key]['raw']
        used_label = np.zeros((data_raw[key]['raw'].shape[0], ), dtype=np.uint8) + idx

        dict_dataset['class'].update({key: idx})
        if idx == 0:
            dict_dataset['peak'] = used_peak
            dict_dataset['raw'] = used_data
            dict_dataset['label'] = used_label
        else:
            dict_dataset['peak'] = np.concatenate((dict_dataset['peak'], used_peak), axis=0)
            dict_dataset['raw'] = np.concatenate((dict_dataset['raw'], used_data), axis=0)
            dict_dataset['label'] = np.concatenate((dict_dataset['label'], used_label), axis=0)
    return dict_dataset


def plot_results(data: dict, take_samples=50) -> None:
    """Plotting the results
    Args:
        data:           Dictionary with spike frames, peak amplitudes and labels
        take_samples:   Only take random N samples from each class
    """
    scale_yval = 32767
    xmid_pos = int(len(data['class']) / 2)
    xstart = 16
    xstop = 80

    # --- Figure #1: Spike Frames
    axs = plt.subplots(1, len(data['class']), sharex=True)[1]
    for key, id in data['class'].items():
        pos_id = np.argwhere(data['label'] == id).flatten()
        pos_random = pos_id[np.random.randint(0, pos_id.size, take_samples)]

        frame_raw = data['raw'][pos_random, xstart:xstop+1]
        frame_mean = np.mean(data['raw'][pos_id, xstart:xstop+1], axis=0)

        axs[id].plot(np.transpose(frame_raw / scale_yval))
        axs[id].plot(frame_mean / scale_yval, 'r')
        axs[id].grid()
        axs[id].set_title(key)

    axs[0].set_ylabel("Spike Norm. Value")
    axs[xmid_pos].set_xlabel("Spike Frame Position")
    plt.subplots_adjust(hspace=0.05, wspace=0.1)

    # --- Figure #2: Histogram - Spike Frame Peak Amplitude
    plt.figure()
    plt.hist(data['peak'], bins=101)
    plt.xlabel('Spike Peak Amplitude [µV]')
    plt.ylabel('Bins')
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    path2save = "C:\\GitHub\\spaike_project\\3_Python\\data\\test"
    load_fzj_onoff_waveforms("C:\\HomeOffice\\Data_Neurosignal\\0A_RGC_FZJ_ONOFF", path2save)

    data = np.load(os.path.join(path2save, "fzj_onoff_waveforms.npy"), allow_pickle=True).flatten()[0]
    plot_results(data)
    print(".done")
