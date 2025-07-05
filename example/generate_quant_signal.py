import numpy as np
import matplotlib.pyplot as plt
from fxpmath import Fxp
from denspp.offline import get_path_to_project
from denspp.offline.plot_helper import save_figure
from denspp.offline.data_generator.waveform_generator import WaveformGenerator


def plot_waveform_types(dataset: dict, show_plot: bool=True) -> None:
    bitwidth = 4
    bitfrac = 3

    sig_used = dataset['sig'] / (dataset['sig'].max() - dataset['sig'].min())
    sig_quant = Fxp(sig_used, signed=True, n_word=bitwidth, n_frac=bitfrac).get_val()

    for idx in range(4):
        fig = plt.figure(figsize=(4, 3))
        if idx == 0:
            plt.plot(dataset['time'], sig_used, color='k', linewidth=2)
        elif idx == 1:
            plt.step(dataset['time'], sig_quant, where='mid', color='k', linewidth=2)
        elif idx == 2:
            plt.stem(dataset['time'], sig_used, linefmt='k', markerfmt='k', basefmt='None')
        else:
            plt.stem(dataset['time'], sig_quant, linefmt='k', markerfmt='k', basefmt='None')

        plt.axis('off')
        plt.tight_layout()
        save_figure(fig, path=f'{get_path_to_project()}/runs', name=f'waveform_type{idx:02d}')

    if show_plot:
        plt.show()


if __name__ == "__main__":
    f_smp = 20
    data0 = WaveformGenerator(
        sampling_rate=f_smp,
        add_noise=False
    ).generate_waveform(
        time_points=[3/f_smp],
        time_duration=[2.],
        waveform_select=['SINE_FULL'],
        polarity_cathodic=[False]
    )
    plot_waveform_types(data0)
