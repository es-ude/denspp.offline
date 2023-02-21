from src.pipeline import PipelineSpike
import matplotlib.pyplot as plt
import numpy as np

def results_afe (signals: PipelineSpike) -> None:
    fs_ana = signals.sample_rate_ana
    fs_adc = signals.sample_rate_adc

    uin = signals.u_in
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.spike_ticks

    tA = np.arange(0, uin.size, 1) / fs_ana
    tD = np.arange(0, xadc.size, 1) / fs_adc

    # --- Plotting
    plt.figure()
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(tA, 1e6* uin, 'k')
    plt.ylabel("$U_{in}$ (µV)")

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(tD, xadc, 'k')
    plt.ylabel("$X_{adc}$")

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(tD, xsda, 'k')
    ax3.plot(tD, xthr, 'r')
    plt.ylabel("$X_{sda}$")

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    idx = 0
    for wave in ticks:
        ax4.plot(tD, 1.25*idx + wave)
        idx += 1

    plt.ylabel("Spike Ticks")
    plt.xlabel("Time t (s)")

def results_fec (signals: PipelineSpike) -> None:
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

    plt.figure()
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(np.transpose(framesIn))

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(np.transpose(framesOut))

    ax3 = plt.subplot(2, 2, 3)
    for idx in range(0, cluster.shape[0]):
        ax3.plot(feat[idx,0], feat[idx, 1], color=color[cluster[idx]], marker='.')

    ax4 = plt.subplot(2, 2, 4)
    for idx in range(0, mean_frames.shape[0]):
        ax4.plot(np.transpose(mean_frames[idx, :]), color=color[idx])


def plot_frames (framesIn: np.ndarray, framesMean: np.ndarray) -> None:
    plt.figure()
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(np.transpose(framesIn))

    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(np.transpose(framesMean))

