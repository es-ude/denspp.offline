import os.path
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import PipelineSpike

def cm_to_inch(value):
    return value/2.54

def save_figure(fig, path: str, name: str):
    format = ['eps', 'svg']
    path2fig = os.path.join(path, name)

    for idx, form in enumerate(format):
        file_name = path2fig + '.' + form
        fig.savefig(file_name, format=form)

def results_afe (signals: PipelineSpike, path: str) -> None:
    fs_ana = signals.fs_ana
    fs_adc = signals.fs_adc

    uin = signals.u_in
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.spike_ticks

    tA = np.arange(0, uin.size, 1) / fs_ana
    tD = np.arange(0, xadc.size, 1) / fs_adc

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(21)))
    plt.subplots_adjust(hspace=0)

    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)

    # Input signal
    ax1.plot(tA, 1e6* uin, 'k')
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
    for idx, wave in enumerate(ticks):
        ax4.plot(tD, 1.25*idx + wave)

    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_transient")


def results_fec(signals: PipelineSpike, path: str) -> None:
    color = ['k', 'r', 'b', 'g']

    if signals.version == 0:
        framesIn = signals.frames_orig
        framesOut = signals.frames_align
    else:
        framesIn = signals.frames_align
        framesOut = signals.frames_denoised

    feat = signals.features
    cluster = signals.cluster_id
    mean_frames = np.zeros(shape=(signals.cluster_no, framesOut.shape[1]))
    mean_cluster = np.zeros(shape=(signals.cluster_no, feat.shape[0]))
    mean_value = np.zeros(shape=(signals.cluster_no, 1))

    idx = 0
    for wave in framesOut:
        mean_frames[cluster[idx], :] += wave
        mean_value[cluster[idx]] += 1
        idx += 1
    mean_frames = mean_frames / mean_value

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.transpose(framesIn))

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(np.transpose(framesOut))

    ax3 = plt.subplot(2, 2, 3)
    for idx in range(0, cluster.shape[0]):
        ax3.plot(feat[idx, 0], feat[idx, 1], color=color[cluster[idx]], marker='.')

    ax4 = plt.subplot(2, 2, 4)
    for idx in range(0, mean_frames.shape[0]):
        ax4.plot(np.transpose(mean_frames[idx, :]), color=color[idx])

    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_fec")

def results_test(signals: PipelineSpike, path: str) -> None:
    feat = signals.features
    x = feat[:, 0]
    y = feat[:, 1]

    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()

    # Create the Axes.
    ax.set(aspect=1)
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

    # Draw the scatter plot and marginals.
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.scatter(x, y)

    bins = 100
    ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal')

def results_paper(signals: PipelineSpike, path: str) -> None:
    color = ['k', 'r', 'b', 'g']
    textsize = 14
    timeCut = [50, 60]

    # --- Selection of Transient signals
    fs_adc = signals.fs_adc
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.spike_ticks
    tD = np.arange(0, xadc.size, 1) / fs_adc

    if signals.version == 0:
        framesOut = signals.frames_align
    else:
        framesOut = signals.frames_denoised

    # --- Selection of FEC signals
    feat = signals.features
    cluster = signals.cluster_id
    mean_frames = np.zeros(shape=(signals.cluster_no, framesOut.shape[1]))
    mean_cluster = np.zeros(shape=(signals.cluster_no, feat.shape[0]))
    mean_value = np.zeros(shape=(signals.cluster_no, 1))

    idx = 0
    for wave in framesOut:
        mean_frames[cluster[idx], :] += wave
        mean_value[cluster[idx]] += 1
        idx += 1
    mean_frames = mean_frames / mean_value

    #plt.figure().set_figwidth(cm_to_inch(16))
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
        ax3.plot(tD, 1 * idx + 0.9 * wave, color=color[idx])

    ax3.set_yticks([0, 1, 2])
    ax3.set_ylabel("Spike Train")
    ax3.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper0")

    # --- Figure 2
    #plt.figure().set_figwidth(cm_to_inch(16))
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
        ax1.plot(feat[idx, 0], feat[idx, 1], color=color[cluster[idx]], marker='.')

    # Spike Frames
    for idx in range(0, mean_frames.shape[0]):
        ax2.plot(np.transpose(mean_frames[idx, :]), color=color[idx])

    ax2.set_xticks([0, 7, 15, 23, 31])
    ax2.set_yticks([-20, 0, 40])
    ax2.set_ylim(-21, 41)
    ax2.set_ylabel('ADC output')
    ax2.set_xlabel('Frame position')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, "pipeline_paper1")

