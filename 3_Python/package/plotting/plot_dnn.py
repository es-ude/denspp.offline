import numpy as np
import matplotlib.pyplot as plt
from package.plotting.plot_common import cm_to_inch, save_figure


def plt_memristor_ref(frames_in: np.ndarray, frames_cluster: np.ndarray, frames_mean: np.ndarray) -> None:
    """Plotting reference signals for testing with BFO memristor-based calculation"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    path = 'runs'
    fig_size =[22, 8]
    textsize = 12
    yrange = [-20, 60]
    error = [-70.3, -111.1, 0.2, 72.85, -119.5]

    use_class = 2
    sel_pos = np.where(frames_cluster == use_class)
    frames_input = frames_in[sel_pos[0], :]

    # --- Plot #1
    plt.figure()
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(cm_to_inch(fig_size[0]), cm_to_inch(fig_size[1]-2)))
    plt.subplots_adjust(hspace=0, wspace=0.5)

    ax = [plt.subplot(1, 5, i+1) for i in range(5)]
    ax[0].set_ylabel("U_top (Sample)")
    for a in ax:
        a.set_ylim(yrange)
        a.plot(np.transpose(frames_input), linewidth=1)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(direction='in')
        a.grid()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if path:
        save_figure(plt, path, "memristor_input")

    # --- Plot #2
    plt.figure()
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(cm_to_inch(fig_size[0]), cm_to_inch(fig_size[1])))
    plt.subplots_adjust(hspace=0, wspace=0.5)

    ax = [plt.subplot(1, 5, i + 1) for i in range(5)]
    ax[0].set_ylabel("U_bot (Ref.)")
    for idx, a in enumerate(ax):
        a.set_ylim(yrange)
        a.plot(frames_mean[idx, :], color=color[idx], linewidth=2)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(direction='in')
        a.grid()
        a.set_title(f'Class {idx}\nDelta = {error[idx]:.1f}')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if path:
        save_figure(plt, path, "memristor_reference")

    plt.show(block=True)

def test_plot(frames_in, frames_cluster):
    cluster_no = np.unique(frames_cluster)

    frames_plot = []
    frames_mean = []
    for idx, clid in enumerate(cluster_no):
        xsel = np.where(frames_cluster == clid)
        frames0 = np.zeros(shape=(len(xsel[0]), frames_in.shape[1]))
        for i, sel in enumerate(xsel[0]):
            frames0[i, :] = frames_in[sel, :]

        frames_plot.append(frames0)
        frames_mean.append(np.mean(frames0, axis=0))

    # --- Plotting
    plt.figure()
    for idx, clid in enumerate(cluster_no):
        # plt.plot(np.transpose(frames_plot[idx]), color='k')
        plt.plot(frames_mean[idx])

    plt.show(block=True)


def results_training(path: str,
                     yin: np.ndarray, ypred: np.ndarray, ymean: np.ndarray,
                     feat: np.ndarray, cluster: np.ndarray,
                     snr: np.ndarray) -> None:
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    textsize = 12

    # --- Pre-Processing
    cluster_no = np.unique(cluster)

    mark_pos = []
    mark_feat0 = []
    mark_feat1 = []
    mark_feat2 = []
    for i, id in enumerate(cluster_no):
        mark_pos.append(np.where(cluster == id))
        mark_feat0.append(feat[mark_pos[-1], 0])
        mark_feat1.append(feat[mark_pos[-1], 1])
        mark_feat2.append(feat[mark_pos[-1], 2])

    # --- Plotting: Statistics
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    ax1.hist(cluster)
    ax1.set_xticks(range(0, np.max(cluster_no)))
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Bins")

    ax2.plot(snr[:, 0], color='k')
    ax2.plot(snr[:, 1], color='r')
    ax2.plot(snr[:, 2], color='g')
    ax2.legend(["min", "mean", "max"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Improved SNR (dB)")

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_hist")

    # --- Plotting: SNR
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)

    plt.plot(snr[:, 0], color='k')
    plt.plot(snr[:, 1], color='r')
    plt.plot(snr[:, 2], color='g')
    plt.legend(["min", "mean", "max"])
    plt.xlabel("Epoch")
    plt.ylabel("Improved SNR (dB)")

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_snr")

    mode = 1
    # --- Plotting - Feat 0
    fig = plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)
    row = 1
    col = 3

    ax1 = plt.subplot(row, col, 1)
    ax1.margins(x=0)
    ax1.set_xticks([0, 7, 15, 23, 31])
    ax2 = plt.subplot(row, col, 2)
    ax2.margins(x=0)
    ax3 = plt.subplot(row, col, 3, sharex=ax1)
    ax3.margins(x=0)

    # Noisy input
    ax1.plot(np.transpose(yin))
    ax1.set_title('Network Input')
    ax1.set_ylabel('Y_in')
    ax1.set_xlabel('Frame position')

    # Feature extraction
    if mode == 0:
        ax2.scatter(feat[:, 0], feat[:, 1])
    else:
        for i, id in enumerate(cluster_no):
            ax2.scatter(mark_feat0[i], mark_feat1[i], color=color[i], marker='.')
    ax2.set_title('Features')
    ax2.set_ylabel('Feat[0]')
    ax2.set_xlabel('Feat[1]')

    # Denoised output
    ax3.plot(np.transpose(ypred))
    if mode == 1:
        for i, id in enumerate(cluster_no):
            ax3.plot(ymean[i, :], color=color[i], linewidth=1.5)
    ax3.set_title('Network Output')
    ax3.set_ylabel('X_pred')
    ax3.set_xlabel('Frame position')

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_out0")

    # --- Plotting - Feat 1
    fig = plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)
    row = 1
    col = 3

    ax1 = plt.subplot(row, col, 1)
    ax1.margins(x=0)
    ax1.set_xticks([0, 7, 15, 23, 31])
    ax2 = plt.subplot(row, col, 2)
    ax2.margins(x=0)
    ax3 = plt.subplot(row, col, 3, sharex=ax1)
    ax3.margins(x=0)

    # Noisy input
    ax1.plot(np.transpose(yin))
    ax1.set_title('Network Input')
    ax1.set_ylabel('Y_in')
    ax1.set_xlabel('Frame position')

    # Feature extraction
    if mode == 0:
        ax2.scatter(feat[:, 0], feat[:, 1])
    else:
        for i, id in enumerate(cluster_no):
            ax2.scatter(mark_feat1[i], mark_feat2[i], color=color[i], marker='.')
    ax2.set_title('Features')
    ax2.set_ylabel('Feat[1]')
    ax2.set_xlabel('Feat[2]')

    # Denoised output
    ax3.plot(np.transpose(ypred))
    if mode == 1:
        for i, id in enumerate(cluster_no):
            ax3.plot(ymean[i, :], color=color[i], linewidth=1.5)
    ax3.set_title('Network Output')
    ax3.set_ylabel('X_pred')
    ax3.set_xlabel('Frame position')

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_out1")

    # --- Plotting - Feat 2
    fig = plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)
    row = 1
    col = 3

    ax1 = plt.subplot(row, col, 1)
    ax1.margins(x=0)
    ax1.set_xticks([0, 7, 15, 23, 31])
    ax2 = plt.subplot(row, col, 2)
    ax2.margins(x=0)
    ax3 = plt.subplot(row, col, 3, sharex=ax1)
    ax3.margins(x=0)

    # Noisy input
    ax1.plot(np.transpose(yin))
    ax1.set_title('Network Input')
    ax1.set_ylabel('Y_in')
    ax1.set_xlabel('Frame position')

    # Feature extraction
    if mode == 0:
        ax2.scatter(feat[:, 0], feat[:, 1])
    else:
        for i, id in enumerate(cluster_no):
            ax2.scatter(mark_feat0[i], mark_feat2[i], color=color[i], marker='.')
    ax2.set_title('Features')
    ax2.set_ylabel('Feat[0]')
    ax2.set_xlabel('Feat[2]')

    # Denoised output
    ax3.plot(np.transpose(ypred))
    if mode == 1:
        for i, id in enumerate(cluster_no):
            ax3.plot(ymean[i, :], color=color[i], linewidth=1.5)
    ax3.set_title('Network Output')
    ax3.set_ylabel('X_pred')
    ax3.set_xlabel('Frame position')

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_out2")

    # --- Plotting: Feature space
    plt.figure()
    ax = plt.axes(projection='3d')

    for i, id in enumerate(cluster_no):
        ax.scatter3D(mark_feat0[i], mark_feat1[i], mark_feat2[i], color=color[i], marker='.')
    ax.set_xlabel('Feat[0]')
    ax.set_ylabel('Feat[1]')
    ax.set_zlabel('Feat[2]')

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path:
        save_figure(plt, path, "ai_training_feat")
