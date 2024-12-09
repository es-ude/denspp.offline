import os
import numpy as np
import h5py
from glob import glob
from datetime import datetime
from tqdm import tqdm
from package.data_process.frame_normalization import DataNormalization
from package.plot.plot_dataset import plot_frames_dataset


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

    data_onoff = {'date_created': datetime.now().strftime("%Y%m%d_%H%M%S"),
                  'sampling_rate_khz': 25, 'electrode': 'mcs60mea'}

    data_temp = dict()
    for file in tqdm(file_list):
        # --- Loading
        hndlr = h5py.File(file, 'r')
        data0 = np.array(hndlr.get('tot_spike_matrix'))
        data1 = np.transpose(data0)
        del data0

        # --- Processing for DataSize reduction
        norm_hndl = DataNormalization("minmax")
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
    data_list = ['OFF sustained', 'OFF transient', 'ON sustained', 'ON transient', 'ONOFF']

    create_time = datetime.now().strftime("%Y-%m-%d")
    dict_dataset = {'data': np.zeros((1, 1)), 'label': np.zeros((1, 1)), 'peak': np.zeros((1, 1)),
                    'dict': data_list, 'create_on': create_time}
    for idx, key in enumerate(data_keys):
        used_peak = data_raw[key]['amp_peak_uV']
        used_data = data_raw[key]['raw']
        used_label = np.zeros((data_raw[key]['raw'].shape[0], ), dtype=np.uint8) + idx

        if idx == 0:
            dict_dataset['peak'] = used_peak
            dict_dataset['data'] = used_data
            dict_dataset['label'] = used_label
        else:
            dict_dataset['peak'] = np.concatenate((dict_dataset['peak'], used_peak), axis=0)
            dict_dataset['data'] = np.concatenate((dict_dataset['data'], used_data), axis=0)
            dict_dataset['label'] = np.concatenate((dict_dataset['label'], used_label), axis=0)
    return dict_dataset


if __name__ == "__main__":
    path2save = './../../data'
    # load_fzj_onoff_waveforms("C:\\HomeOffice\\Data_Neurosignal\\0A_RGC_FZJ_ONOFF", path2save)

    data = np.load(os.path.join(path2save, "2024-11-25_Dataset_RGC_MCS_ONOFF_FZJ.npy"), allow_pickle=True).flatten()[0]
    plot_frames_dataset(data)
    print(".done")
