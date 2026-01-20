import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from denspp.offline import get_path_to_project
from denspp.offline.plot_helper import save_figure, get_plot_color, get_textsize_paper
from denspp.offline.preprocessing import SettingsFilter, Filtering, FilterCoeffs


def plot_filter_fir_coeffs(coeffs: FilterCoeffs, path2save: Path=Path("."), block_plot:bool=False) -> None:
    fig = plt.figure()

    plt.stem(coeffs.b, linefmt="black", markerfmt="ko", basefmt="gray", label="Coeff b")

    plt.xlabel('Coeff. Index', size=get_textsize_paper())
    plt.ylabel('Coeff. Value', size=get_textsize_paper())

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlim([0, len(coeffs.b)-1])
    plt.grid()
    plt.tight_layout()
    if path2save:
        save_figure(fig, str(path2save), "filter_coeffs")
    if block_plot:
        plt.show(block=True)


if __name__ == "__main__":
    path = get_path_to_project("runs")
    bitwidth_total = 8
    bitwidth_frac = 6
    sets_filt = SettingsFilter(
        gain=1,
        fs=2e3,
        n_order=1,
        f_filt=[80.],
        type='iir',
        f_type='butter',
        b_type='lowpass'
    )
    dut = Filtering(setting=sets_filt)

    coeff_qnt = dut.get_coeffs_quantized(bit_size=bitwidth_total, bit_frac=bitwidth_frac)[0]

    dut.plot_freq_response(
        num_points=201,
        show_plot=False,
        path2save=path
    )
    plot_filter_fir_coeffs(
        coeffs=coeff_qnt, #dut.get_coeffs(),
        path2save=Path(path),
        block_plot=True
    )
