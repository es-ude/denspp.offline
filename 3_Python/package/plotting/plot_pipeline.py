import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from package.plotting.plot_common import cm_to_inch, save_figure

from package.template.pipeline_signals import PipelineSignal
from package.nsp import calc_amplitude, calc_autocorrelogram, calc_firing_rate


color_none = ['#929591']
color_cluster = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
text_size = 14


def results_afe0(signals: PipelineSignal, no_electrode: int, path="", time_cut=[]) -> None:
    """Plotting the results in type of ... """
    fs_ana = signals.fs_ana
    uin = signals.u_in
    tA = np.arange(0, uin.size, 1) / fs_ana

    # --- Plot afe
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)

    ax1.plot(tA, 1e6 * uin, 'k')
    ax1.set_ylabel("U_in [µV]")
    ax2.plot(tA, 1e6 * signals.u_pre, color='k', drawstyle='steps-post')
    ax2.set_ylabel("U_pre [µV]")
    ax3.plot(tA, 1e6 * signals.u_quant, color='k', drawstyle='steps-post')
    ax3.set_ylabel("U_adc [µV]")
    ax3.set_xlabel("Time t (s)")

    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
        addon_zoom = '_zoom'
    else:
        ax1.set_xlim([tA[0], tA[-1]])
        addon_zoom = ''

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_afe_elec" + str(no_electrode) + addon_zoom)


def results_afe1(signals: PipelineSignal, no_electrode: int, path="", time_cut=[]) -> None:
    """Plotting the results in type of ... """
    fs_ana = signals.fs_ana
    fs_adc = signals.fs_adc
    fs_dig = signals.fs_dig

    uin = signals.u_in
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.frames_align[1]
    ticks_id = signals.frames_align[2]
    cluster = np.unique(ticks_id)

    tA = np.arange(0, uin.size, 1) / fs_ana
    tD = np.arange(0, xadc.size, 1) / fs_dig

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    ax1 = plt.subplot(411)
    ax2 = plt.subplot(412, sharex=ax1)
    ax3 = plt.subplot(413, sharex=ax1)
    ax4 = plt.subplot(414, sharex=ax1)

    # Input signal
    ax1.plot(tA, 1e6 * uin, 'k')
    ax1.set_ylabel("$U_{in}$ (µV)")

    # ADC output
    ax2.plot(tD, xadc, color='k', drawstyle='steps-post')
    ax2.set_ylabel("$X_{adc}$")

    # SDA + Thresholding
    ax3.plot(tD, xsda, color='k', drawstyle='steps-post')
    ax3.plot(tD, xthr, color='r', drawstyle='steps-post')
    ax3.set_ylabel("$X_{sda}$")

    # Spike Ticks
    ax4.set_ylabel("Spike Ticks")
    ax4.set_xlabel("Time t (s)")
    for id in cluster:
        sel_x = np.where(ticks_id == id)[0]
        sel_ticks = ticks[sel_x]
        ax4.eventplot(positions=tD[sel_ticks], orientation="horizontal",
                      lineoffsets=0.45+id, linelengths=0.9,
                      color=color_cluster[id])
    ax4.set_ylim([cluster[0], 1+cluster[-1]])

    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
        addon_zoom = '_zoom'
    else:
        ax1.set_xlim([tD[0], tD[-1]])
        addon_zoom = ''

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient_elec" + str(no_electrode) + addon_zoom)


def results_afe2(signals: PipelineSignal, no_electrode: int, path="", time_cut=[], show_noise=False):
    """Plotting ADC output"""
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
        sel = [tick-12, tick+30]
        time0.append(time[tick_old:sel[0]])
        time0.append(time[sel[0]:sel[1]])
        tran0.append(xadc[tick_old:sel[0]] if show_noise else np.zeros(shape=(len(xadc[tick_old:sel[0]]), ), dtype=int))
        tran0.append(xadc[sel[0]:sel[1]])
        colo0.append(color_none[0])
        colo0.append(color_cluster[ticks_id[idx]])

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

    # plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient_sorted" + str(no_electrode) + addon_zoom)


def results_fec(signals: PipelineSignal, no_electrode: int, path="") -> None:
    """Plotting results """
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
        ax3.plot(feat[idx, 0], feat[idx, 1], color=color_cluster[id], marker='.')

    ax4.set_title("Mean Frames (Clustered)")
    for idx, frame in enumerate(frames_mean):
        ax4.plot(np.transpose(frame), color=color_cluster[idx],
                 marker='.', markersize=4, drawstyle='steps-post')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fec_elec" + str(no_electrode))


# TODO: ConfusionMatrix erstellen
def results_confusion(true_labels, pred_labels, no_electrode: int, path="") -> None:
    title = "Spike Sorting"

    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=true_labels,
        y_pred=pred_labels,
        cmap=plt.cm.Blues,
        normalize='true',
        colorbar=True,
        )

    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
    plt.show()

    if path:
            save_figure(plt, path, "pipeline_ivt" + str(no_electrode))


def results_ivt(signals: PipelineSignal, no_electrode: int, path="") -> None:
    """Plotting the results of interval timing spikes of each cluster"""
    frames = signals.frames_align[0]
    cluster = signals.frames_align[2]
    cluster_num = np.unique(cluster)
    mean_frames = np.zeros(shape=(len(cluster), frames.shape[1]))
    for idx, id in enumerate(cluster_num):
        x0 = np.where(cluster == id)[0]
        mean_frames[idx, :] = np.mean(frames[x0], axis=0)

    its = calc_firing_rate(signals.spike_ticks, signals.fs_dig)

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
        axs[val_plot+0].plot(mean_frames[id, :], color=color_cluster[id], drawstyle='steps-post')
        axs[val_plot+1].hist(scale * its[id], bins=no_bins)

    axs[0].set_xticks([0, 7, 15, 23, 31])
    axs[0].set_ylabel("ADC output")
    axs[0].set_xlabel("Frame position")

    axs[1].set_xlim(0, 1000)
    axs[1].set_ylabel("No. bins")
    axs[1].set_xlabel("Interval timing [ms]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_ivt" + str(no_electrode))


def results_correlogram(signals: PipelineSignal, no_electrode: int, path="") -> None:
    """Plotting the results of interval timing spikes of each cluster"""
    val_in = calc_autocorrelogram(signals.spike_ticks, signals.fs_dig)
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
        save_figure(plt, path, "pipeline_correlogram" + str(no_electrode))


def results_firing_rate(signals: PipelineSignal, no_electrode: int, path="") -> None:
    fr_in = calc_firing_rate(signals.spike_ticks, signals.fs_dig)
    no_cluster = len(fr_in)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    axs = list()
    for idx in range(0, no_cluster):
        axs.append(plt.subplot(no_cluster, 1, idx+1))

    for idx, ax in enumerate(axs):
        ax.plot(fr_in[idx][0, :], fr_in[idx][1, :], color=color_cluster[idx], drawstyle='steps-post')

    axs[no_cluster-1].set_xlabel("Time t [s]")
    axs[0].set_ylabel("Firing rate [Spikes/s]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fr" + str(no_electrode))


def results_cluster_amplitude(signals: PipelineSignal, no_electrode: int, path="") -> None:
    amp = calc_amplitude(signals.frames_align)
    cluster = signals.frames_align[2]
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
        time = np.array(time) / signals.fs_dig
        amp_min = np.array(amp_min)
        amp_max = np.array(amp_max)
        axs[sel+0].plot(time, amp_min, color=color_cluster[idx], marker='.', linestyle='None')
        axs[sel+1].plot(time, amp_max, color=color_cluster[idx], marker='.', linestyle='None')

    axs[0].set_ylabel('Min. amp')
    axs[1].set_ylabel('Max. amp')
    axs[1].set_xlabel('Time [s]')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_amp" + str(no_electrode))


def results_paper(signals: PipelineSignal, no_electrode: int, path="", time_cut=[]) -> None:
    """Plotting results of end-to-end spike sorting for paper"""
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
    plt.rcParams.update({'font.size': text_size})
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)

    ax1.plot(tD, xadc, color='k', drawstyle='steps-post')
    ax1.set_yticks([-20, 0, 40])
    ax1.set_ylim(-21, 41)
    ax1.set_ylabel("ADC output")
    ax1.xaxis.set_visible(False)
    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
    else:
        ax1.set_xlim([tD[0], tD[-1]])

    # Spike detection and thresholding
    ax2.plot(tD, xsda, color='k', drawstyle='steps-post')
    ax2.plot(tD, xthr, color='r', drawstyle='steps-post')
    ax2.xaxis.set_visible(False)
    ax2.set_ylabel("SDA output")

    # Spike ticks
    for id in cluster:
        sel_x = np.where(ticks_id == id)[0]
        sel_ticks = ticks[sel_x]
        ax3.eventplot(positions=tD[sel_ticks], orientation="horizontal",
                      lineoffsets=0.45+id, linelengths=0.9,
                      color=color_cluster[id])

    ax3.set_ylim([cluster[0], 1+cluster[-1]])
    ax3.set_ylabel("Spike Train")
    ax3.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper0_elec" + str(no_electrode))

    # --- Figure 2
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(6)))
    plt.rcParams.update({'font.size': text_size})
    plt.subplots_adjust(wspace=0.4)
    ax1 = plt.subplot(121)
    ax1.margins(x=0)
    ax2 = plt.subplot(122)
    ax2.margins(x=0)

    # Clustered Features
    ax1.set_ylabel('Feat. 1')
    ax1.set_xlabel('Feat. 2')
    for idx in range(0, cluster.shape[0]):
        ax1.plot(feat[idx, 0], feat[idx, 1], color=color_cluster[cluster[idx]], marker='.', drawstyle='steps-post')

    # Spike Frames
    for idx in range(0, mean_frames.shape[0]):
        ax2.plot(np.transpose(mean_frames[idx, :]), color=color_cluster[idx],
                 marker='.', markersize=4, drawstyle='steps-post')

    ax2.set_xticks([0, 7, 15, 23, 31])
    ax2.set_yticks([-20, 0, 40])
    ax2.set_ylim(-21, 41)
    ax2.set_ylabel('ADC output')
    ax2.set_xlabel('Frame position')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper1_elec" + str(no_electrode))
