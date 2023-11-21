import numpy as np
import matplotlib.pyplot as plt
from package.plotting.plot_common import cm_to_inch, save_figure


def plt_memristor_ref(frames_in: np.ndarray, frames_cluster: np.ndarray, frames_mean: np.ndarray) -> None:
    """Plotting reference signals for testing with BFO memristor-based calculation"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    path = 'runs'
    fig_size = [22, 8]
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


def results_training(path: str,
                     yin: np.ndarray, ypred: np.ndarray, ymean: np.ndarray,
                     feat: np.ndarray, cluster: np.ndarray,
                     snr: np.ndarray | list, xframes=50, num_feat=3) -> None:
    data_labeled = True

    # --- Pre-Processing
    cluster_no = np.unique(cluster)
    mark_feat = [[] for idx in range(0, num_feat)]
    take_frames = list()
    for i, id in enumerate(cluster_no):
        pos = np.where(cluster == id)[0]
        # Take only X frames per cluster
        np.random.shuffle(pos)
        take_frames.append(pos[:xframes])
        # Separating the features for plotting
        for idx in range(0, num_feat):
            mark_feat[idx].append(feat[pos, idx])

    # --- Plotting: Inference model
    plot_autoencoder_results(
        mark_feat, [0, 1], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, path2save=path
    )
    plot_autoencoder_results(
        mark_feat, [0, 2], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, path2save=path
    )
    plot_autoencoder_results(
        mark_feat, [1, 2], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, path2save=path
    )

    # --- Plotting: Feature Space and Metrics
    plot_autoencoder_snr(snr, path)
    plot_autoencoder_features(cluster_no, mark_feat, [0, 1, 2], path)


def plot_autoencoder_snr(snr: list, path2save='') -> None:
    """"""
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 12})
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid()

    if snr[0].size == 3:
        snr0 = np.array(snr, dtype=float)
        plt.plot(snr0[:, 0], color='k', marker='.', label='min')
        plt.plot(snr0[:, 1], color='r', marker='.', label='mean')
        plt.plot(snr0[:, 2], color='g', marker='.', label='max')
        plt.legend()
        plt.xticks(np.linspace(0, snr0.shape[0], num=7, endpoint=True))
    else:
        plt.boxplot(snr, patch_artist=True, showfliers=False)
        pos = np.linspace(1, len(snr), num=10, endpoint=True)
        plt.xticks(pos)

    plt.xlabel("Epoch")
    plt.ylabel("Improved SNR (dB)")

    plt.tight_layout(pad=0.5)
    if path2save:
        save_figure(plt, path2save, "ai_training_snr")


def plot_autoencoder_features(cluster_no: np.ndarray, mark_feat: list, idx: [0, 1, 2], path2save='') -> None:
    """"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

    plt.figure(figsize=(cm_to_inch(12), cm_to_inch(9)))
    plt.rcParams.update({'font.size': 12})
    ax = plt.axes(projection='3d')

    for i, id in enumerate(cluster_no):
        ax.scatter3D(mark_feat[idx[0]][i], mark_feat[idx[1]][i], mark_feat[idx[2]][i], color=color[i], marker='.')
    ax.set_xlabel('Feat[0]')
    ax.set_ylabel('Feat[1]')
    ax.set_zlabel('Feat[2]')

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "ai_training_feat")


def plot_autoencoder_results(mark_feat: list, mark_idx: list,
                             frames_in: np.ndarray, frames_out: np.ndarray, frames_mean: np.ndarray,
                             cluster_no: np.ndarray, take_frames: list, data_labeled=False, path2save='') -> None:
    """Handler for plotting results"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 12})
    plt.subplots_adjust(hspace=0, wspace=0.5)
    plt.grid(visible=True)
    row = 1
    col = 3

    axs = list()
    for idx in range(0, row * col):
        axs.append(plt.subplot(row, col, 1+idx))

    # Noisy input
    for pos in take_frames:
        axs[0].plot(np.transpose(frames_in[pos, :]), linewidth=0.5)
    axs[0].set_title('Input')
    axs[0].set_xlabel('Frame position')
    axs[0].set_xticks(np.linspace(0, frames_in.shape[1]-1, num=6, endpoint=True, dtype=int))

    # Feature extraction
    for i, id in enumerate(cluster_no):
        axs[1].scatter(mark_feat[mark_idx[0]][i], mark_feat[mark_idx[1]][i], color=color[i], marker='.')
    axs[1].set_title('Feature Space')
    axs[1].set_ylabel(f'Feat[{mark_idx[0]}]')
    axs[1].set_xlabel(f'Feat[{mark_idx[1]}]')

    # Denoised output
    for pos in take_frames:
        axs[2].plot(np.transpose(frames_out[pos, :]), linewidth=0.5)
    if data_labeled:
        for i, id in enumerate(cluster_no):
            axs[2].plot(frames_mean[i, :], color=color[i], linewidth=1.5)
    axs[2].set_title('Output')
    axs[2].set_xlabel('Frame position')
    axs[2].set_xticks(np.linspace(0, frames_mean.shape[1]-1, num=6, endpoint=True, dtype=int))

    for ax in axs:
        ax.grid()
        ax.margins(x=0)

    plt.tight_layout(pad=0.5)
    # --- Saving plots
    if path2save:
        save_figure(plt, path2save, f"ai_training_out{mark_idx[0]}-{mark_idx[1]}")


def plot_statistic_data(train_cl: np.ndarray | list, valid_cl=None, path2save='', cl_dict=None) -> None:
    """Plotting the statistics of the data"""
    do_plots_avai = isinstance(valid_cl, np.ndarray | list)
    dict_available = isinstance(cl_dict, list)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 12})
    plt.subplots_adjust(hspace=0, wspace=0.5)
    axs = list()
    for idx in range(0, 1+do_plots_avai):
        axs.append(plt.subplot(1, 1 + do_plots_avai, 1+idx))

    # Histogram of Training data
    check = np.unique(train_cl, return_counts=True)
    xbins = check[0].tolist()
    xbins.append(check[0].max()+1)

    axs[0].hist(train_cl, bins=xbins, align='left', rwidth=0.8, color='k')
    axs[0].set_xticks(xbins[:-1])
    if dict_available:
        axs[0].set_xticklabels(cl_dict if check[0].size != 1 else [cl_dict[0]])

    axs[0].set_ylabel("Bins")
    axs[0].set_ylim([int(0.99*check[1].min()), int(1.01*check[1].max())])
    axs[0].set_title('Training')

    # Histogram of Validation data
    if do_plots_avai:
        check = np.unique(valid_cl, return_counts=True)
        xbins = check[0].tolist()
        xbins.append(check[0].max() + 1)

        axs[1].hist(valid_cl, bins=xbins, align='left', stacked=True, rwidth=0.8, color='r')
        axs[1].set_xticks(xbins[:-1])
        if dict_available:
            axs[1].set_xticklabels(cl_dict if check[0].size != 1 else [cl_dict[0]])

        axs[1].set_ylim([int(0.99 * check[1].min()), int(1.01 * check[1].max())])
        axs[1].set_title('Validation')

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Cluster")

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "ai_training_histdata")
