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


def results_input(signals: list) -> None:
    uin = signals[0]
    tA = np.linspace(0, 1, uin.shape[0])

    # --- Plot afe
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    ax0 = plt.subplot(331)
    ax1 = plt.subplot(332, sharex=ax0)
    ax2 = plt.subplot(333, sharex=ax0)
    ax3 = plt.subplot(334, sharex=ax0)
    ax4 = plt.subplot(335, sharex=ax0)
    ax5 = plt.subplot(336, sharex=ax0)
    ax6 = plt.subplot(337, sharex=ax0)
    ax7 = plt.subplot(338, sharex=ax0)
    ax8 = plt.subplot(339, sharex=ax0)

    ax0.plot(tA, uin[:, 0], 'k')
    ax1.plot(tA, uin[:, 1], 'k')
    ax2.plot(tA, uin[:, 2], 'k')
    ax3.plot(tA, uin[:, 3], 'k')
    ax4.plot(tA, uin[:, 4], 'k')
    ax5.plot(tA, uin[:, 5], 'k')
    ax6.plot(tA, uin[:, 6], 'k')
    ax7.plot(tA, uin[:, 7], 'k')
    ax8.plot(tA, uin[:, 8], 'k')

    ax7.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    #if path:
        #save_figure(plt, path, "pipeline_afe_elec" + str(no_electrode))
