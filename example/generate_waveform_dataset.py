import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.logger import define_logger_runtime
from denspp.offline.plot_helper import get_plot_color
from denspp.offline.data_generator.waveform_dataset import build_waveform_dataset, SettingsWaveformDataset


if __name__ == "__main__":
    define_logger_runtime(False)

    settings = SettingsWaveformDataset(
        wfg_type=['RECT', 'SAW_NEG', 'LIN_FALL', 'GAUSS'],
        wfg_freq=[2.0, 2.0, 2.0, 2.0],
        num_samples=1,
        time_idle=0.05,
        scale_amp=1.0,
        sampling_rate=10e3,
        noise_add=False,
        noise_pwr_db=-30.0,
        do_normalize=False
    )
    dataset = build_waveform_dataset(settings_data=settings)
    signal_type = dataset['dict']

    # --- Define plots
    plt.figure()
    axs = [plt.subplot(len(signal_type), 1, idx + 1) for idx in range(0, len(signal_type))]

    for idx, (data, label) in enumerate(zip(dataset['data'], dataset['label'])):
        label_text = signal_type[label]
        time0 = np.linspace(0, data.shape[0], data.shape[0]) / settings.sampling_rate
        axs[idx].plot(time0, data, color=get_plot_color(idx), label=f"{label_text}")
        axs[idx].legend()
        axs[idx].grid()
        axs[idx].set_ylabel(r"$V_{sig}(t)$ / V")

    axs[-1].set_xlabel(r"Time $t$ / s")
    plt.tight_layout()
    plt.show(block=True)
