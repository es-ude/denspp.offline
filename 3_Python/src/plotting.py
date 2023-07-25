import os.path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from src.pipeline_signals import PipelineSignal

# TODO: Plots für IVT und FR anpassen in Abhängigkeit der Cluster-Anzahl
# TODO: Schöne Rasterplots erstellen (Problem: Ungleiche Vektorlänge)

color_cluster = ['k', 'r', 'b', 'g', 'y', 'c', 'm']


def cm_to_inch(value):
    return value / 2.54


def save_figure(fig, path: str, name: str):
    format = ['eps', 'svg']
    path2fig = os.path.join(path, name)

    for idx, form in enumerate(format):
        file_name = path2fig + '.' + form
        fig.savefig(file_name, format=form)


def results_afe0(signals: PipelineSignal, path: str, no_electrode: int) -> None:
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
    ax2.plot(tA, 1e6 * signals.u_chp, 'k')
    ax2.set_ylabel("U_chopper [µV]")
    ax3.plot(tA, 1e6 * signals.u_pre, 'k')
    ax3.set_ylabel("Output pre-amplifier [µV]")
    ax3.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_afe_elec" + str(no_electrode))


def results_afe1(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    fs_ana = signals.fs_ana
    fs_adc = signals.fs_adc

    uin = signals.u_in
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr

    if signals.spike_ticks is None:
        testin = np.zeros(shape=xadc.shape)
        testin[signals.x_pos] = 1
        ticks = list()
        ticks.append(testin)
    else:
        ticks = signals.x_pos

    tA = np.arange(0, uin.size, 1) / fs_ana
    tD = np.arange(0, xadc.size, 1) / fs_adc

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
    ax2.plot(tD, xadc, 'k')
    ax2.set_ylabel("$X_{adc}$")

    # SDA + Thresholding
    ax3.plot(tD, xsda, 'k')
    ax3.plot(tD, xthr, 'r')
    ax3.set_ylabel("$X_{sda}$")

    # Spike Ticks
    ax4.set_ylabel("Spike Ticks")
    ax4.set_xlabel("Time t (s)")
    # ax4.eventplot(positions=tD[ticks], orientation="horizontal", lineoffsets=0, linelengths=0.1, color=color_cluster[:no_cluster])
    for idx, wave in enumerate(ticks):
        ax4.plot(tD, 1.25 * idx + wave, color=color_cluster[idx])

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient_elec" + str(no_electrode))


def results_fec(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    """Plotting results """
    framesIn = signals.frames_orig
    framesOut = signals.frames_align

    feat = signals.features
    cluster = signals.cluster_id
    mean_frames = np.zeros(shape=(signals.cluster_no, framesOut.shape[1]))
    for idx, id in enumerate(np.unique(cluster)):
        x0 = np.where(cluster == id)[0]
        mean_frames[idx, :] = np.mean(framesOut[x0], axis=0)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    ax1 = plt.subplot(221)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224, sharex=ax2)

    ax1.plot(np.transpose(framesIn))
    ax2.plot(np.transpose(framesOut))
    for idx in range(0, cluster.shape[0]):
        ax3.plot(feat[idx, 0], feat[idx, 1], color=color_cluster[cluster[idx]], marker='.')

    for idx in range(0, mean_frames.shape[0]):
        ax4.plot(np.transpose(mean_frames[idx, :]), color=color_cluster[idx])

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fec_elec" + str(no_electrode))


def results_ivt(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    """Plotting the results of interval timing spikes of each cluster"""
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))

    ax1 = plt.subplot(231)
    ax2 = plt.subplot(232, sharex=ax1)
    ax3 = plt.subplot(233, sharex=ax1)
    ax4 = plt.subplot(234)
    ax5 = plt.subplot(235, sharex=ax4)
    ax6 = plt.subplot(236, sharex=ax4)

    # --- Mean waveform
    frames = signals.frames_align
    cluster = signals.cluster_id
    mean_frames = np.zeros(shape=(signals.cluster_no, frames.shape[1]))
    for idx, id in enumerate(np.unique(cluster)):
        x0 = np.where(cluster == id)[0]
        mean_frames[idx, :] = np.mean(frames[x0], axis=0)

    ax1.plot(mean_frames[0, :], color=color_cluster[0])
    ax2.plot(mean_frames[1, :], color=color_cluster[1])
    ax3.plot(mean_frames[2, :], color=color_cluster[2])
    ax1.set_xticks([0, 7, 15, 23, 31])
    ax1.set_ylabel("ADC output")
    ax2.set_xlabel("Frame position")

    # --- Histograms
    scale = 1e3
    ax4.hist(scale * signals.its[0], bins=100)
    ax5.hist(scale * signals.its[1], bins=100)
    ax6.hist(scale * signals.its[2], bins=100)
    ax4.set_xlim(0, 1000)
    ax4.set_ylabel("No. bins")
    ax5.set_xlabel("Interval timing [ms]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_ivt" + str(no_electrode))


def results_confusion(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    do_norm = True
    title = "Spike Sorting"
    x_in = 0
    x_out = 0

    disp = ConfusionMatrixDisplay.from_estimator(
        x=x_in,
        y=x_out,
        cmap=plt.cm.Blues,
        normalize=do_norm,
        colorbar=True
    )
    disp.ax_.set_title(title)

    if path:
        save_figure(plt, path, "pipeline_ivt" + str(no_electrode))


def results_correlogram(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    """Plotting the results of interval timing spikes of each cluster"""
    val_in = signals.correlogram
    no_cluster = signals.cluster_no
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    ax4 = plt.subplot(131)
    ax5 = plt.subplot(132, sharex=ax4)
    ax6 = plt.subplot(133, sharex=ax4)

    scale = 1e3
    no_bins_sel = 200

    ax4.hist(scale * val_in[0], bins=no_bins_sel)
    ax5.hist(scale * val_in[1], bins=no_bins_sel)
    ax6.hist(scale * val_in[2], bins=no_bins_sel)

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_correlogram" + str(no_electrode))


def results_firing_rate(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    fr_in = signals.firing_rate

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132, sharex=ax1)
    ax3 = plt.subplot(133, sharex=ax1)

    ax1.plot(fr_in[0][0, :], fr_in[0][1, :], color=color_cluster[0])
    ax2.plot(fr_in[1][0, :], fr_in[1][1, :], color=color_cluster[1])
    ax3.plot(fr_in[2][0, :], fr_in[2][1, :], color=color_cluster[2])

    ax2.set_xlabel("Time t [s]")
    ax1.set_ylabel("Firing rate [Spikes/s]")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fr" + str(no_electrode))


def results_paper(signals: PipelineSignal, path: str, no_electrode: int) -> None:
    """Plotting results of end-to-end spike sorting for paper"""
    textsize = 14
    timeCut = [50, 60]

    # --- Selection of Transient signals
    fs_adc = signals.fs_adc
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.spike_ticks
    tD = np.arange(0, xadc.size, 1) / fs_adc
    framesOut = signals.frames_align

    # --- Selection of FEC signals
    feat = signals.features
    cluster = signals.cluster_id
    mean_frames = np.zeros(shape=(signals.cluster_no, framesOut.shape[1]))
    for idx, id in enumerate(np.unique(cluster)):
        x0 = np.where(cluster == id)[0]
        mean_frames[idx, :] = np.mean(framesOut[x0], axis=0)

    # plt.figure().set_figwidth(cm_to_inch(16))
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(12)))
    plt.rcParams.update({'font.size': textsize})
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)

    # --- Plotting results
    # ADC output
    ax1.plot(tD, xadc, 'k')
    ax1.set_yticks([-20, 0, 40])
    ax1.set_ylim(-21, 41)
    ax1.set_ylabel("ADC output")
    ax1.xaxis.set_visible(False)
    plt.xlim(timeCut)

    # Spike detection and thresholding
    ax2.plot(tD, xsda, 'k')
    ax2.plot(tD, xthr, 'r')
    ax2.xaxis.set_visible(False)
    ax2.set_ylabel("SDA output")

    # Spike ticks
    for idx, wave in enumerate(ticks):
        ax3.plot(tD, 1 * idx + 0.9 * wave, color=color_cluster[idx])

    ax3.set_yticks([0, 1, 2])
    ax3.set_ylabel("Spike Train")
    ax3.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper0_elec" + str(no_electrode))

    # --- Figure 2
    # plt.figure().set_figwidth(cm_to_inch(16))
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(6)))
    plt.rcParams.update({'font.size': textsize})
    plt.subplots_adjust(wspace=0.4)
    ax1 = plt.subplot(121)
    ax1.margins(x=0)
    ax2 = plt.subplot(122)
    ax2.margins(x=0)

    # Clustered Features
    ax1.set_ylabel('Feat. 1')
    ax1.set_xlabel('Feat. 2')
    for idx in range(0, cluster.shape[0]):
        ax1.plot(feat[idx, 0], feat[idx, 1], color=color_cluster[cluster[idx]], marker='.')

    # Spike Frames
    for idx in range(0, mean_frames.shape[0]):
        ax2.plot(np.transpose(mean_frames[idx, :]), color=color_cluster[idx])

    ax2.set_xticks([0, 7, 15, 23, 31])
    ax2.set_yticks([-20, 0, 40])
    ax2.set_ylim(-21, 41)
    ax2.set_ylabel('ADC output')
    ax2.set_xlabel('Frame position')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper1_elec" + str(no_electrode))
