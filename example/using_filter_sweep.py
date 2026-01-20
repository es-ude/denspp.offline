import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from denspp.offline import get_path_to_project
from denspp.offline.plot_helper import save_figure, get_plot_color, get_textsize_paper
from denspp.offline.preprocessing import SettingsFilter, Filtering


def plot_filter_coeffs(coeffs: list, fsweep: np.ndarray, path2save: Path=Path("."), block_plot:bool=False) -> None:
    coeffa = list()
    coeffb = list()
    for coeff in coeffs:
        coeffa.append(coeff.a)
        coeffb.append(coeff.b)

    data_a = np.array(coeffa)
    data_b = np.array(coeffb)
    fig = plt.figure()

    plt.plot(fsweep, data_a)
    plt.plot(fsweep, data_b)

    plt.xlabel('Cutoff Frequency / Hz', size=get_textsize_paper())
    plt.ylabel('Coeff. Value', size=get_textsize_paper())

    ax = plt.gca()
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    plt.xlim([fsweep[0], 1e3]) #fsweep[-1]])
    plt.grid()
    plt.tight_layout()
    if path2save:
        save_figure(fig, str(path2save), "filter_coeffs_sweep")
    if block_plot:
        plt.show(block=True)


if __name__ == "__main__":
    path = get_path_to_project("runs")
    bitwidth_total = 8
    bitwidth_frac = 6
    sets_filt = SettingsFilter(
        gain=1,
        fs=2e3,
        n_order=201,
        f_filt=[80.],
        type='fir',
        f_type='butter',
        b_type='lowpass'
    )

    coeffs = list()
    fsweep = np.linspace(start=1, stop=sets_filt.fs/2, num=101, endpoint=False)
    for val in fsweep:
        sets_filt.f_filt = [val]
        coeffs.append(Filtering(setting=sets_filt).get_coeffs())

    plot_filter_coeffs(
        coeffs=coeffs,
        fsweep=fsweep,
        path2save=Path(path),
        block_plot=True
    )
