import numpy as np
from torch.utils.data import Dataset
from package.stim.waveform_generator import WaveformGenerator
from package.analog.dev_noise import noise_awgn


class DatasetWFG(Dataset):
    """Dataset for defining different waveforms for training"""
    def __init__(self, fs: float, waveforms: np.ndarray, classes: np.ndarray,
                 cl_names: list, do_autoencoder=False) -> None:
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


def generate_dataset(selected_wfg: list, num_wfg_class: int, freq_wfg: float, sampling_rate: float,
                     adding_noise=False, pwr_noise_db=-30.0, get_info_classes=False,
                     do_normalize_rms=False) -> DatasetWFG:
    """Generating dataset
    Args:
        selected_wfg:       Selected types of waveforms
        num_wfg_class:      Number of samples for each waveform class
        freq_wfg:           Frequency of the waveform
        sampling_rate:      Sampling rate
        adding_noise:       Adding noise to output
        get_info_classes:   Getting print output with available signal types
        do_normalize_rms:   Normalizing the energy of waveform in order to have similar true RMS
    Returns:
        Dataset with waveforms sample inside
    """
    # --- Define classes
    wfg_generator = WaveformGenerator(sampling_rate)
    wfg_dict = wfg_generator.get_dictionary_classes()

    if get_info_classes:
        print("\nGetting information about signal types")
        print("\n====================================================")
        for idx, id in enumerate(wfg_dict):
            print(f"Class {idx:02d} = {id}")
        print("====================================================")

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
            t0, signal0, rms = wfg_generator.generate_waveform(t_start, t_wfg, [sel_wfg], [False])
            if adding_noise:
                signal0 += noise_awgn(t0.size, sampling_rate, pwr_noise_db)

            waveforms_signals[idx * num_wfg_class + num_ite, :] = signal0 if not do_normalize_rms else 1 / rms * signal0
            waveforms_classes[idx * num_wfg_class + num_ite] = idx
            waveforms_rms[idx * num_wfg_class + num_ite] = rms

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    num_wfg_class = 1
    sel_wfg_class = [0, 5, 7, 8]

    dataset = generate_dataset(sel_wfg_class, num_wfg_class, 2, 10e3,
                               get_info_classes=True, adding_noise=True, pwr_noise_db=-30)
    signal_type = dataset.class_names

    # --- Define plots
    plt.figure()
    axs = [plt.subplot(len(dataset), 1, idx + 1) for idx in range(0, len(dataset))]

    color_plot = 'krym'
    for idx, data in enumerate(dataset):
        time0 = np.linspace(0, data[0].shape[0], data[0].shape[0]) / dataset.sampling_rate
        axs[idx].plot(time0, data[0], color=color_plot[idx], label=f"{signal_type[data[1]]}")
        axs[idx].legend()
        axs[idx].grid()

    plt.tight_layout()
    plt.show()
