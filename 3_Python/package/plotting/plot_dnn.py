import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from package.plotting.plot_common import cm_to_inch, save_figure


def results_training(path: str,
                     yin: np.ndarray, ypred: np.ndarray, ymean: np.ndarray,
                     feat: np.ndarray, yclus: np.ndarray, snr: list,
                     cl_dict=None, xframes=50, num_feat=3) -> None:
    data_labeled = True

    # --- Pre-Processing
    yclus0 = yclus.flatten() if len(yclus.shape) == 2 else yclus
    cluster_no = np.unique(yclus0)
    mark_feat = [[] for idx in range(0, num_feat)]
    take_frames = list()
    for i, id in enumerate(cluster_no):
        pos = np.where(yclus0 == id)[0]
        # Take only X frames per cluster
        np.random.shuffle(pos)
        take_frames.append(pos[:xframes])
        # Separating the features for plotting
        for idx in range(0, num_feat):
            mark_feat[idx].append(feat[pos, idx])

    # --- Plotting: Inference model
    plot_autoencoder_run(
        mark_feat, [0, 1], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, data_classname=cl_dict, path2save=path
    )
    plot_autoencoder_run(
        mark_feat, [0, 2], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, data_classname=cl_dict, path2save=path
    )
    plot_autoencoder_run(
        mark_feat, [1, 2], yin, ypred, ymean,
        cluster_no, take_frames, data_labeled=data_labeled, data_classname=cl_dict, path2save=path
    )

    # --- Plotting: Feature Space and Metrics
    plot_autoencoder_snr(snr, path)
    plot_autoencoder_features(cluster_no, mark_feat, [0, 1, 2], data_classname=cl_dict, path2save=path)


def plot_autoencoder_snr(snr: list, path2save='', do_boxplot=False) -> None:
    """Plotting the Signal-to-Noise Ratio (SNR) over the epochs"""
    # --- Processing
    snr_processed = list()
    for snr_fold in snr:
        if not do_boxplot:
            snr0 = np.zeros(shape=(len(snr_fold), 3), dtype=float)
            for idx, snr_epoch in enumerate(snr_fold):
                snr0[idx, :] = snr_epoch.min(), np.median(snr_epoch), snr_epoch.max()
            snr_processed.append(snr0)
        else:
            snr_processed.append(snr_fold)

    # --- Plotting
    for idx, snr0 in enumerate(snr_processed):
        plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
        plt.rcParams.update({'font.size': 12})
        plt.subplots_adjust(hspace=0, wspace=0.5)
        plt.grid()

        if not do_boxplot:
            plt.plot(snr0[:, 0], color='k', marker='.', label='min')
            plt.plot(snr0[:, 1], color='r', marker='.', label='mean')
            plt.plot(snr0[:, 2], color='g', marker='.', label='max')
            plt.legend()
            pos = np.linspace(0, snr0.shape[0]-1, num=11, endpoint=True, dtype=int)
        else:
            plt.boxplot(snr, patch_artist=True, showfliers=False)
            pos = np.linspace(0, len(snr0)-1, num=11, endpoint=True, dtype=int)

        plt.xticks(pos)
        plt.xlim([pos[0]-1, pos[-1]+1])
        plt.xlabel("Epoch")
        plt.ylabel("Improved SNR (dB)")

        plt.tight_layout(pad=0.5)
        if path2save:
            save_figure(plt, path2save, f"ai_training_snr_fold{idx:03d}")


def plot_autoencoder_features(cluster_no: np.ndarray, mark_feat: list, idx: [0, 1, 2], data_classname=None, path2save='') -> None:
    """Plotting the feature space of the autoencoder"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

    fig = plt.figure(figsize=(cm_to_inch(12), cm_to_inch(9)))
    plt.rcParams.update({'font.size': 6})
    Axes3D(fig)
    ax = plt.axes(projection='3d')

    for i, id in enumerate(cluster_no):
        ax.scatter3D(mark_feat[idx[0]][i], mark_feat[idx[1]][i], mark_feat[idx[2]][i], color=color[i], marker='.')
    ax.set_xlabel('Feat[0]')
    ax.set_ylabel('Feat[1]')
    ax.set_zlabel('Feat[2]')
    if isinstance(data_classname, list):
        if not len(data_classname) == 0:
            ax.legend(data_classname)

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "ai_training_feat")


def plot_autoencoder_run(mark_feat: list, mark_idx: list,
                         frames_in: np.ndarray, frames_out: np.ndarray, frames_mean: np.ndarray,
                         cluster_no: np.ndarray, take_frames: list,
                         data_classname=None, data_labeled=False, path2save='') -> None:
    """Plotting the autoencoder in-/output for an inference"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 6})
    plt.subplots_adjust(hspace=0, wspace=0.5)
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
    if isinstance(data_classname, list):
        if not len(data_classname) == 0:
            axs[1].legend(data_classname)

    # Denoised output
    if data_labeled:
        for i, id in enumerate(cluster_no):
            axs[2].plot(frames_mean[i, :], color=color[i], linewidth=2)
    for pos in take_frames:
        axs[2].plot(np.transpose(frames_out[pos, :]), linewidth=0.5)

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
        if not len(cl_dict) == 0:
            use_cl_dict = list()
            for idx in np.unique(train_cl):
                use_cl_dict.append(cl_dict[int(idx)])
            axs[0].set_xticklabels(use_cl_dict if check[0].size != 1 else [use_cl_dict[0]])

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
            if not len(cl_dict) == 0:
                use_cl_dict = list()
                for idx in np.unique(train_cl):
                    use_cl_dict.append(cl_dict[int(idx)])
                axs[1].set_xticklabels(use_cl_dict if check[0].size != 1 else [use_cl_dict[0]])

        axs[1].set_ylim([int(0.99 * check[1].min()), int(1.01 * check[1].max())])
        axs[1].set_title('Validation')

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Cluster")

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "ai_training_histdata")
