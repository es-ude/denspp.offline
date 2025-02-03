import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
from .waveform_generator import WaveformGenerator
from denspp.offline.analog.dev_noise import SettingsNoise, RecommendedSettingsNoise


class DatasetWFG(Dataset):
    def __init__(self, fs: float, waveforms: np.ndarray, classes: np.ndarray,
                 cl_names: list, do_autoencoder=False) -> None:
        """Dataset for defining different waveforms for training"""
        self.sampling_rate = fs
        self.__signals = waveforms
        self.__classes = classes
        self.class_names = cl_names
        self.__do_autoencoder = do_autoencoder

    def __len__(self) -> int:
        return self.__signals.shape[0]

    def __getitem__(self, idx) -> [np.ndarray, np.ndarray]:
        if self.__do_autoencoder:
            return self.__signals[idx, :], self.__signals[idx, :]
        else:
            return self.__signals[idx, :], self.__classes[idx]


def generate_dataset(selected_wfg: list, num_wfg_class: int, freq_wfg: float, sampling_rate: float, scale_amp:float=1.0,
                     adding_noise: bool=False, pwr_noise_db:float=-30.0, get_info_classes:bool=False, do_normalize_rms:bool=False,
                     settings_noise: SettingsNoise=RecommendedSettingsNoise) -> DatasetWFG:
    """Generating dataset with waveforms
    Args:
        selected_wfg:       Selected types of waveforms
        num_wfg_class:      Number of samples for each waveform class
        freq_wfg:           Frequency of the waveform
        sampling_rate:      Sampling rate
        scale_amp:          Scaling factor for waveform amplitude
        adding_noise:       Adding noise to output
        pwr_noise_db:       PWR noise in dB
        get_info_classes:   Getting print output with available signal types
        do_normalize_rms:   Normalizing the energy of waveform in order to have similar true RMS
        settings_noise:     Dataclass for handling the noise behaviour
    Returns:
        Dataset with waveforms sample inside
    """
    # --- Define classes
    settings0 = deepcopy(settings_noise)
    settings0.wgn_dB = pwr_noise_db
    wfg_generator = WaveformGenerator(
        sampling_rate=sampling_rate,
        add_noise=adding_noise,
        settings_noise=settings_noise
    )
    wfg_dict = wfg_generator.get_dictionary_classes(get_info_classes)

    t_window = 1.5 / freq_wfg
    t_wfg = [1 / freq_wfg]
    t_start = [0.25 / freq_wfg]
    num_samples = len(selected_wfg) * num_wfg_class

    # --- Generation of signal
    waveforms_signals = np.zeros((num_samples, int(sampling_rate * t_window)), dtype=float)
    waveforms_classes = np.zeros((num_samples, ), dtype=int)
    waveforms_rms = np.zeros(num_samples, )
    for idx, sel_wfg in enumerate(selected_wfg):
        for num_ite in range(0, num_wfg_class):
            waveform = wfg_generator.generate_waveform(t_start, t_wfg, [sel_wfg], [False])

            signal = scale_amp * waveform['sig'] if not do_normalize_rms else scale_amp / waveform['rms'] * waveform['sig']
            waveforms_signals[idx * num_wfg_class + num_ite, :] = signal
            waveforms_classes[idx * num_wfg_class + num_ite] = idx
            waveforms_rms[idx * num_wfg_class + num_ite] = waveform['rms']

    # --- Do energy normalization (Only check RMS values)
    if do_normalize_rms:
        rms_classes = np.zeros(len(selected_wfg), )
        for i, id in enumerate(np.unique(selected_wfg)):
            pos = np.argwhere(waveforms_classes == i)
            rms_classes[i] = np.mean(waveforms_rms[pos])

    # --- Getting dictionary of signal type
    waveforms_dict = list()
    for id in np.unique(selected_wfg):
        waveforms_dict.append(wfg_dict[id])

    return DatasetWFG(
        fs=sampling_rate,
        waveforms=waveforms_signals,
        classes=waveforms_classes,
        cl_names=waveforms_dict
    )
