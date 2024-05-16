import os.path
import numpy as np
import matplotlib.pyplot as plt

color_cluster = ['k', 'r', 'b', 'g', 'y', 'c', 'm']


def cm_to_inch(value):
    return value / 2.54


def save_figure(fig, path: str, name: str):
    format = ['eps', 'svg']
    path2fig = os.path.join(path, name)

    for idx, form in enumerate(format):
        file_name = path2fig + '.' + form
        fig.savefig(file_name, format=form)


def results_input(signals: list, fs: float | int, label_xpos=(), label_id=()) -> None:
    """"""
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)
    label_avai = not len(label_xpos) == 0

    # Plot definition
    axs = list()
    for idx in range(len(signals)):
        if idx == 0:
            axs.append(plt.subplot(3, 3, idx+1))
        else:
            axs.append(plt.subplot(3, 3, idx+1, sharex=axs[0]))

    color_label = ['r', 'b', 'y', 'm', 'k']
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
    #if path:
        #save_figure(plt, path, "pipeline_afe_elec" + str(no_electrode))
