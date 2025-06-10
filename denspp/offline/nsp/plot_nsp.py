import numpy as np
from matplotlib import pyplot as plt
from denspp.offline.nsp import calc_firing_rate, calc_autocorrelogram, calc_amplitude
from denspp.offline.plot_helper import cm_to_inch, get_plot_color, save_figure


def plot_nsp_ivt(signals: dict, no_electrode: int, path: str="", show_plot: bool=False) -> None:
    """Plotting the results of interval timing spikes of each cluster
    :param signals:         Pipeline signal object
    :param no_electrode:    Number of electrodes to plot
    :param path:            Path to save figure
    :param show_plot:       If true, show plot
    :return:                None
    """
    frames = signals["frames_align"][0]
    cluster = signals["frames_align"][2]
    cluster_num = np.unique(cluster)
    mean_frames = np.zeros(shape=(len(cluster), frames.shape[1]))
    for idx, id in enumerate(cluster_num):
        x0 = np.where(cluster == id)[0]
        mean_frames[idx, :] = np.mean(frames[x0], axis=0)

    its = calc_firing_rate(signals["spike_ticks"], signals["fs_dig"])

    scale = 1e3
    no_bins = 100

    # Plotting
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    plt.subplots_adjust(hspace=0)
    axs = list()
    for idx in range(0, len(cluster_num)):
        # Plots for mean waveform (ungerade) and hists (gerade)
        if idx == 0:
            axs.append(plt.subplot(2, len(cluster_num), idx+1))
            axs.append(plt.subplot(2, len(cluster_num), idx+1 + len(cluster_num)))
        else:
            axs.append(plt.subplot(2, len(cluster_num), idx+1, sharex=axs[0]))
            axs.append(plt.subplot(2, len(cluster_num), idx+1 + len(cluster_num), sharex=axs[1]))

    for idx, id in enumerate(cluster_num):
        val_plot = 2*idx
        axs[val_plot+0].plot(mean_frames[id, :], color=get_plot_color(int(id)), drawstyle='steps-post')
        axs[val_plot+1].hist(scale * its[int(id)], bins=no_bins)

    axs[0].set_xticks([0, 7, 15, 23, 31])
    axs[0].set_ylabel("ADC output")
    axs[0].set_xlabel("Frame position")

    axs[1].set_xlim(0, 1000)
    axs[1].set_ylabel("No. bins")
    axs[1].set_xlabel("Interval timing [ms]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "nsp_pipeline_ivt" + str(no_electrode))
    if show_plot:
        plt.show(block=True)


def plot_nsp_correlogram(signals: dict, no_electrode: int, path: str="", show_plot: bool=False) -> None:
    """Plotting the results of interval timing spikes of each cluster
    :param signals:         Pipeline signal object
    :param no_electrode:    Number of electrodes to plot
    :param path:            Path to save figure
    :param show_plot:       If true, show plot
    :return:                None
    """
    val_in = calc_autocorrelogram(signals["spike_ticks"], signals["fs_dig"])
    cluster_num = len(val_in)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))

    axs = list()
    for idx, val in enumerate(val_in):
        if idx == 0:
            axs.append(plt.subplot(cluster_num, 1, idx+1))
        else:
            axs.append(plt.subplot(cluster_num, 1, idx+1, sharex=axs[0]))

    scale = 1e3
    no_bins_sel = 200

    for idx, val in enumerate(val_in):
        axs[idx].hist(scale * val, bins=no_bins_sel)

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "nsp_pipeline_correlogram" + str(no_electrode))
    if show_plot:
        plt.show(block=True)


def plot_firing_rate(signals: dict, no_electrode: int, path: str="", show_plot: bool=False) -> None:
    """Function for plotting the firing rate of choicen electrode
    :param signals:         Pipeline signal object
    :param no_electrode:    Number of electrodes to plot
    :param path:            Path to save figure
    :param show_plot:       If true, show plot
    :return:                None
    """
    fr_in = calc_firing_rate(signals["spike_ticks"], signals["fs_dig"])
    no_cluster = len(fr_in)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    axs = list()
    for idx in range(0, no_cluster):
        axs.append(plt.subplot(no_cluster, 1, idx+1))

    for idx, ax in enumerate(axs):
        ax.plot(fr_in[idx][0, :], fr_in[idx][1, :], color=get_plot_color(idx), drawstyle='steps-post')

    axs[no_cluster-1].set_xlabel("Time t [s]")
    axs[0].set_ylabel("Firing rate [Spikes/s]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "nsp_pipeline_fr" + str(no_electrode))
    if show_plot:
        plt.show(block=True)


def plot_nsp_cluster_amplitude(signals: dict, no_electrode: int, path: str="", show_plot: bool=False) -> None:
    """Function for plotting the spike frame amplitude values of each amplitude during time
    :param signals:         Pipeline signal object
    :param no_electrode:    Number of electrodes to plot
    :param path:            Path to save figure
    :param show_plot:       If true, show plot
    :return:                None
    """
    amp = calc_amplitude(signals["frames_align"])
    cluster = signals["frames_align"][2]
    cluster_no = np.unique(cluster)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    axs = list()
    for idx, id in enumerate(cluster_no):
        if idx == 0:
            axs.append(plt.subplot(2, len(cluster_no), idx+1))
            axs.append(plt.subplot(2, len(cluster_no), idx+1 + len(cluster_no)))
        else:
            axs.append(plt.subplot(2, len(cluster_no), idx+1, sharex=axs[0]))
            axs.append(plt.subplot(2, len(cluster_no), idx+1 + len(cluster_no), sharex=axs[0]))

    for idx, amp0 in enumerate(amp):
        sel = 2*idx
        time = list()
        amp_min = list()
        amp_max = list()
        for val in amp0:
            time.append(val[0])
            amp_min.append(val[1])
            amp_max.append(val[2])
        time = np.array(time) / signals["fs_dig"]
        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)
        axs[sel+0].plot(time, amp_min, color=get_plot_color(idx), marker='.', linestyle='None')
        axs[sel+1].plot(time, amp_max, color=get_plot_color(idx), marker='.', linestyle='None')

    axs[0].set_ylabel('Min. amp')
    axs[1].set_ylabel('Max. amp')
    axs[1].set_xlabel('Time [s]')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "nsp_pipeline_amp" + str(no_electrode))
    if show_plot:
        plt.show(block=True)
