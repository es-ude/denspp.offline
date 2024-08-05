import numpy as np
import matplotlib.pyplot as plt
from package.plot.plot_common import _cm_to_inch, _save_figure

color_label = ['r', 'b', 'y', 'm', 'k']


def results_input(signals: list, fs: float | int, label_xpos=(), label_id=(), path2save='', addon='') -> None:
    """"""
    plt.figure(figsize=(_cm_to_inch(16), _cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)
    label_avai = not len(label_xpos) == 0

    # Plot definition
    axs = list()
    for idx in range(len(signals)):
        if idx == 0:
            axs.append(plt.subplot(3, 3, idx+1))
        else:
            axs.append(plt.subplot(3, 3, idx+1, sharex=axs[0]))

    for idx, ax in enumerate(axs):
        if label_avai:
            for id in np.unique(label_id):
                if not id == -1:
                    xpos_sel = np.argwhere(label_id == id).flatten()
                    ylabel = 100 + np.ones(xpos_sel.shape)
                    ax.plot(label_xpos[idx][xpos_sel] / fs, ylabel, marker=".", markersize=12,
                            linestyle="None", color=color_label[id])
        tA = np.linspace(0, signals[idx].shape[0], signals[idx].shape[0]) / fs
        ax.plot(tA, signals[idx], 'k')

    axs[7].set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path2save:
        _save_figure(plt, path2save, f"pipeline_emg_elec{addon}")
