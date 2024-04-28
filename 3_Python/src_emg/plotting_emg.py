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


def results_input(signals: list, fs: float | int) -> None:
    tA = np.linspace(0, signals[-1].shape[0] / fs, signals[-1].shape[0])

    # --- Plot afe
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    # Plot definition
    axs = list()
    for idx in range(len(signals)-1):
        if idx == 0:
            axs.append(plt.subplot(3, 3, idx+1))
        else:
            axs.append(plt.subplot(3, 3, idx+1, sharex=axs[0]))

    for idx, ax in enumerate(axs):
        ax.plot(tA, signals[idx], 'k')

    axs[7].set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    #if path:
        #save_figure(plt, path, "pipeline_afe_elec" + str(no_electrode))
