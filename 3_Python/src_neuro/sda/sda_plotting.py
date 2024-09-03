import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable

from package.plot.plot_common import save_figure, scale_auto_value


def plot_histogramm(in0: np.ndarray, in1: np.ndarray, in2: np.ndarray, path2save='', show_plot=False) -> None:
    plt.figure()
    ax0 = plt.subplot(131)
    ax1 = plt.subplot(132)
    ax2 = plt.subplot(133)

    ax0.hist(in0)
    ax1.hist(in1)
    ax2.hist(in2)

    plt.tight_layout()
    # --- Saving plots
    if path2save:
        save_figure(plt, path2save, f"histogram_transient")
    if show_plot:
        plt.show(block=True)


def plot_results_single(time: np.ndarray, spk_signal: np.ndarray, spk_sda: np.ndarray, spk_thr: np.ndarray,
                        spk_frame: np.ndarray, methods: str, path="", show_plot=False) -> None:
    """Plotting the results of a single run"""
    plt.figure()
    plt.subplots_adjust(hspace=0)

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(325)
    ax4 = plt.subplot(326)

    scaley, unity = scale_auto_value(spk_signal)
    ax1.set_title(methods)
    ax1.plot(time, scaley * spk_signal, color='k')
    ax1.set_ylabel(f'U_elec [{unity}V]')
    ax1.set_xlim([0, time[-1]])

    ax2.plot(time, spk_sda, color='k')
    ax2.plot(time, spk_thr, color='r')
    ax2.set_ylabel('X_sda,thr')
    ax1.set_xlabel('Time [s]')

    ax3.plot(np.transpose(spk_frame), color='k')
    ax3.plot(np.mean(spk_frame, axis=0), color='r')
    ax3.set_ylabel('X_frame')
    ax3.set_xlabel('Position')
    ax3.set_xlim([0, spk_frame.shape[1]-1])

    ax4.hist(scaley * spk_signal, bins=1000, color='k')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, f"{methods}_transient")
    if show_plot:
        plt.show(block=True)


def plot_results_sweep(spk_firing_rate: np.ndarray, snr: np.ndarray, dt_acc: np.ndarray, tr_rate: np.ndarray,
                       fr_rate: np.ndarray, methods: str, path="", show_plot=False) -> None:
    """Plotting the results of a single run"""
    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0)
    axs = [plt.subplot(1, 3, i+1) for i in range(3)]

    levels = 100 * np.array([0.25, 0.5, 0.75, 0.95])
    fontsize = 8
    vmin = 0
    vmax = 100
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    X, Y = np.meshgrid(snr, spk_firing_rate)
    cpf = axs[0].contourf(X, Y, 100 * dt_acc, len(levels), vmin=vmin, vmax=vmax)
    line_colors = ['black' for l in cpf.levels]
    cp = axs[0].contour(X, Y, 100 * dt_acc, levels=levels, colors='black', vmin=vmin, vmax=vmax)
    axs[0].clabel(cp, fontsize=fontsize, colors=line_colors)
    cpf.set_norm(norm)

    cpf = axs[1].contourf(X, Y, 100 * tr_rate, len(levels), vmin=vmin, vmax=vmax)
    line_colors = ['black' for l in cpf.levels]
    cp = axs[1].contour(X, Y, 100 * tr_rate, levels=levels, colors=line_colors, vmin=vmin, vmax=vmax)
    axs[1].clabel(cp, fontsize=fontsize, colors=line_colors)
    cpf.set_norm(norm)

    cpf = axs[2].contourf(X, Y, 100 * fr_rate, len(levels), vmin=vmin, vmax=vmax)
    line_colors = ['black' for l in cpf.levels]
    cp = axs[2].contour(X, Y, 100 * fr_rate, levels=levels, colors=line_colors, vmin=vmin, vmax=vmax)
    axs[2].clabel(cp, fontsize=fontsize, colors=line_colors)
    cpf.set_norm(norm)

    fig.colorbar(ScalarMappable(norm=norm))

    plt.suptitle(methods)
    axs[0].set_title("Accuracy [%]")
    axs[0].set_xlabel("SNR [dB]")
    axs[0].set_ylabel("Firing Rate [Hz]")
    axs[0].set_xticks(np.linspace(-18, 12, 6))
    axs[1].set_title("Precision (TPR) [%]")
    axs[1].set_xlabel("SNR [dB]")
    axs[2].set_title("Recall (FPR) [%]")
    axs[2].set_xlabel("SNR [dB]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, f"{methods}_sweep2D")
    if show_plot:
        plt.show(block=True)
