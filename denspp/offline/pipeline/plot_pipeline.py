import numpy as np
import matplotlib.pyplot as plt

from denspp.offline.pipeline.pipeline_signal import PipelineSignal
from denspp.offline.plot_helper import cm_to_inch, save_figure, get_plot_color, get_textsize_paper, get_plot_color_inactive


def plot_pipeline_afe(signals: PipelineSignal, no_electrode: int, path="", time_cut=(),
                      show_plot=False) -> None:
    """Plotting the pipeline results of the front-end device
    :param signals:         the pipeline signals (class PipelineSignal)
    :param no_electrode:    the number of electrodes to plot
    :param path:            the path of the output file
    :param time_cut:        the time cutoff to plot
    :param show_plot:       If true, show the plot
    :return:                None
    """
    fs_ana = signals.fs_ana
    fs_dig = signals.fs_dig

    uin = signals.u_in
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.frames_align[1]
    ticks_id = signals.frames_align[2]
    cluster = np.unique(ticks_id)

    time_ana = np.arange(0, uin.size, 1) / fs_ana
    time_dig = np.arange(0, xadc.size, 1) / fs_dig

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)

    # Input signal
    ax1.plot(time_ana, 1e6 * uin, 'k')
    ax1.set_ylabel("$U_{in}$ (µV)")

    # ADC output
    ax2.plot(time_dig, xadc, color='k', drawstyle='steps-post')
    ax2.set_ylabel("$X_{adc}$")

    # SDA + Thresholding
    ax3.plot(time_dig, xsda, color='k', drawstyle='steps-post')
    ax3.plot(time_dig, xthr, color='r', drawstyle='steps-post')
    ax3.set_ylabel("$X_{sda}$")

    # Spike Ticks
    ax4.set_ylabel("Spike Ticks")
    ax4.set_xlabel("Time t (s)")
    for id in cluster:
        sel_x = np.where(ticks_id == id)[0]
        sel_ticks = ticks[sel_x]
        ax4.eventplot(positions=time_dig[sel_ticks], orientation="horizontal",
                      lineoffsets=0.45+id, linelengths=0.9,
                      color=get_plot_color(id))
    ax4.set_ylim([cluster[0], 1+cluster[-1]])

    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
        addon_zoom = '_zoom'
    else:
        ax1.set_xlim([time_dig[0], time_dig[-1]])
        addon_zoom = ''

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient_elec" + str(no_electrode) + addon_zoom)
    if show_plot:
        plt.show(block=True)


def plot_transient_highlight_spikes(signals: PipelineSignal, no_electrode: int, path="", time_cut=(),
                                    show_noise=False, show_plot=False) -> None:
    """Plotting the detected spike activity from transient data (highlighted, noise in gray)
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param path:            Path to save the figures
    :param show_plot:       If true, show plot
    :param show_noise:      If true, show noise (otherwise flat line)
    :return:                None
    """
    fs_dig = signals.fs_dig
    xadc = signals.x_adc
    time = np.arange(0, xadc.size, 1) / fs_dig
    ticks = signals.frames_align[1]
    ticks_id = signals.frames_align[2]

    time0 = list()
    tran0 = list()
    colo0 = list()
    tick_old = 0
    for idx, tick in enumerate(ticks):
        sel = [int(tick)-12, int(tick)+30]
        time0.append(time[tick_old:sel[0]])
        time0.append(time[sel[0]:sel[1]])
        tran0.append(xadc[tick_old:sel[0]] if show_noise else np.zeros(shape=(len(xadc[tick_old:sel[0]]), ), dtype=int))
        tran0.append(xadc[sel[0]:sel[1]])
        colo0.append(get_plot_color_inactive())
        colo0.append(get_plot_color(ticks_id[idx]))
        tick_old = sel[1]

    # --- Plot generation
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    # plt.subplots_adjust(hspace=0)
    axs = list()
    for idx in range(0, 1):
        axs.append(plt.subplot(1, 2, 1+2*idx))
        axs.append(plt.subplot(1, 2, 2+2*idx, sharey=axs[2*idx]))

    # Subplot 1: Transient signal (colored)
    for idx, time1 in enumerate(time0):
        axs[0].plot(time1, tran0[idx], linewidth=1, color=colo0[idx], drawstyle='steps-post')

    # --- Subplot 2: Histogram (from Subplot 1)
    no_bins = 1 + abs(max(xadc)) + abs(min(xadc))
    if not len(time_cut) == 0:
        sel0 = np.where(time >= time_cut[0])[0][0]
        sel1 = np.where(time >= time_cut[1])[0][0] -1
        x_bins = xadc[sel0:sel1]
    else:
        x_bins = xadc
    x_nonzero = np.where(x_bins != 0)[0]
    axs[1].hist(xadc[x_nonzero], color='k',
                density=True, log=True,
                bins=no_bins,
                orientation="horizontal")

    # --- Axis test
    axs[0].set_xlabel('Time t [ms]')
    axs[0].set_ylabel('X_adc(t) [ ]')
    axs[0].grid()

    axs[1].set_xlabel('Density')
    axs[1].grid()

    # --- Zooming
    if not len(time_cut) == 0:
        axs[0].set_xlim(time_cut)
        addon_zoom = '_zoom'
    else:
        axs[0].set_xlim([time[0], time[-1]])
        addon_zoom = ''

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient_sorted" + str(no_electrode) + addon_zoom)
    if show_plot:
        plt.show(block=True)


def plot_pipeline_frame_sorted(signals: PipelineSignal, no_electrode: int,
                               path="", show_plot=False) -> None:
    """Plotting the detected spike frame activity of used transient dataset
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param path:            Path to save the figures
    :param show_plot:       If true, show plot
    :return:                None
    """
    frames_in = signals.frames_orig[0]
    frames_out = signals.frames_align[0]
    feat = signals.features
    cluster = signals.frames_align[2]
    cluster_no = len(np.unique(cluster))

    frames_mean = np.zeros(shape=(cluster_no, frames_out.shape[1]))
    for idx, id in enumerate(np.unique(cluster)):
        x0 = np.where(cluster == id)[0]
        frames_mean[idx, :] = np.mean(frames_out[x0], axis=0)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224, sharex=ax2)

    ax1.set_title("Input Frames")
    ax1.plot(np.transpose(frames_in), marker='.', markersize=4, drawstyle='steps-post')

    ax2.set_title("Aligned Frames")
    ax2.plot(np.transpose(frames_out), marker='.', markersize=4, drawstyle='steps-post')

    ax3.set_title("Feature Space")
    for idx, id in enumerate(cluster):
        ax3.plot(feat[idx, 0], feat[idx, 1], color=get_plot_color(id), marker='.')

    ax4.set_title("Mean Frames (Clustered)")
    for idx, frame in enumerate(frames_mean):
        ax4.plot(np.transpose(frame), color=get_plot_color(idx),
                 marker='.', markersize=4, drawstyle='steps-post')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fec_elec" + str(no_electrode))
    if show_plot:
        plt.show(block=True)


def plot_pipeline_results(signals: PipelineSignal, no_electrode: int, path="", time_cut=(), show_plot=False) -> None:
    """Plotting results of end-to-end spike sorting for paper
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param path:            Path to save the figures
    :param time_cut:        Time cut
    :param show_plot:       If true, show plot
    :return:                None
    """
    # --- Selection of Transient signals
    fs_adc = signals.fs_dig
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    used_frames = signals.frames_align

    # --- Selection of FEC signals
    tD = np.arange(0, xadc.size, 1) / fs_adc
    frames_out = used_frames[0]
    ticks = used_frames[1]
    ticks_id = used_frames[2]
    feat = signals.features
    cluster = np.unique(ticks_id)
    mean_frames = np.zeros(shape=(len(cluster), frames_out.shape[1]))
    for idx, id in enumerate(cluster):
        x0 = np.where(ticks_id == id)[0]
        mean_frames[idx, :] = np.mean(frames_out[x0], axis=0)

    # --- Plot 1: Transient signals
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(12)))
    plt.rcParams.update({'font.size': get_textsize_paper()})
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    ax1.plot(tD, xadc, color='k', drawstyle='steps-post')
    ax1.set_ylabel("ADC output")
    ax1.xaxis.set_visible(False)
    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
    else:
        ax1.set_xlim([tD[0], tD[-1]])
    ax1.set_xlabel("Time t (s)")

    # Spike ticks
    for id in cluster:
        sel_x = np.where(ticks_id == id)[0]
        sel_ticks = ticks[sel_x]
        ax2.eventplot(positions=tD[sel_ticks], orientation="horizontal",
                      lineoffsets=0.45+id, linelengths=0.9,
                      color=get_plot_color(id))

    ax2.set_ylim([cluster[0], 1+cluster[-1]])
    ax2.set_ylabel("Spike Train")
    ax2.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper0_elec" + str(no_electrode))

    # --- Figure 2
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(6)))
    plt.rcParams.update({'font.size': get_textsize_paper()})
    plt.subplots_adjust(wspace=0.4)
    ax1 = plt.subplot(121)
    ax1.margins(x=0)
    ax2 = plt.subplot(122)
    ax2.margins(x=0)

    # Clustered Features
    ax1.set_ylabel('Feat. 1')
    ax1.set_xlabel('Feat. 2')
    for idx in range(0, cluster.shape[0]):
        ax1.plot(feat[idx, 0], feat[idx, 1], color=get_plot_color(int(cluster[idx])),
                 marker='.', drawstyle='steps-post')

    # Spike Frames
    for idx in range(0, mean_frames.shape[0]):
        ax2.plot(np.transpose(mean_frames[idx, :]), color=get_plot_color(idx),
                 marker='.', markersize=4, drawstyle='steps-post')

    ax2.set_ylabel('ADC output')
    ax2.set_xlabel('Frame position')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper1_elec" + str(no_electrode))
    if show_plot:
        plt.show(block=True)


def plot_signals_neural_cluster(dataset, path2save='', show_plot=False) -> None:
    """Plotting the mean waveforms of each cluster directly from dataset
    :param dataset:     Training Dataset containing rawdata (spike activity)
    :param path2save:   path to save the figure
    :param show_plot:   show plot
    :return:            None
    """
    num_cl = dataset.__frames_me.shape[0]
    num_frame_size = dataset.__frames_me.shape[1] - 1

    plt.figure()
    axs = [plt.subplot(2, int(np.ceil(num_cl/2)), 1+idx) for idx in range(num_cl)]
    for idx, frame in enumerate(dataset.__frames_me):
        axs[idx].plot(frame, color=get_plot_color(idx))
        axs[idx].grid()
        axs[idx].set_title(dataset.__labeled_dictionary[idx])
        axs[idx].set_xlim(0, num_frame_size)
        axs[idx].set_xticks(np.linspace(0, num_frame_size, 5, dtype=int))

    plt.tight_layout(pad=0.5)
    if path2save:
        save_figure(plt, path2save, f"pipeline_neural_cluster")
    if show_plot:
        plt.show(block=True)
