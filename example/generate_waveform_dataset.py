import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.data_generator.waveform_dataset import generate_dataset


if __name__ == "__main__":

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
