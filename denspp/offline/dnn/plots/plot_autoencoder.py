import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from denspp.offline.plot_helper import (
    cm_to_inch,
    save_figure,
    get_plot_color
)


def results_autoencoder_training(
        path: str,
        yin: np.ndarray, ypred: np.ndarray, ymean: np.ndarray,
        feat: np.ndarray, yclus: np.ndarray,
        cl_dict=None, xframes: int=50, num_feat: int=3, show_plot: bool=False
) -> None:
    """Plotting results from Autoencoder Training for Neural Spike Sorting
    Args:
        path:       Path for saving the results of plots
        yin:        Input signal with neural spike frames
        ypred:      Predicted classes
        ymean:      Mean waveform of all spike classes
        feat:       Numpy array with features for plotting feature space
        yclus:      Numpy array with cluster results
        snr:        List with SNR values
        cl_dict:    Dict with class labels
        xframes:    Size of spike frames
        num_feat:   Number of features
        show_plot:  Showing plots [Default: False]
    Returns:
        None
    """
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
    plot_3d_featspace(
        yclus0, mark_feat, [0, 1, 2], data_classname=cl_dict, path2save=path,
        show_plot=show_plot
    )


def plot_3d_featspace(
        labels: np.ndarray, mark_feat: list,
        idx: list=(0, 1, 2), data_classname=None,
        path2save: str='', show_plot: bool=False, show_ticks: bool=False
) -> None:
    """Plotting the feature space of the autoencoder
    :param labels:          Numpy array with labels of the dataset
    :param mark_feat:       Numpy array with extracted mark features of the dataset
    :param idx:             Numpy array with used indices of the feature space (mark_feat)
    :param data_classname:  Numpy array with used label names
    :param path2save:       Path to save the figure
    :param show_plot:       If true, show plot
    :param show_ticks:      If true, show ticks
    :return:                None
    """
    fig = plt.figure(figsize=(cm_to_inch(14), cm_to_inch(10)))
    Axes3D(fig)
    ax = plt.axes(projection='3d')
    fontsize_label = 12

    cluster_no = np.unique(labels)
    for i, id in enumerate(cluster_no):
        ax.scatter3D(mark_feat[idx[0]][i], mark_feat[idx[1]][i], mark_feat[idx[2]][i],
                     color=get_plot_color(i), marker='.')
    ax.set_xlabel('Feat[0]', fontsize=fontsize_label)
    ax.set_ylabel('Feat[1]', fontsize=fontsize_label)
    ax.set_zlabel('Feat[2]', fontsize=fontsize_label)
    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    if isinstance(data_classname, list):
        if not len(data_classname) == 0:
            ax.legend(data_classname)

    plt.tight_layout()
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "ai_training_feat")
    if show_plot:
        plt.show(block=True)


def plot_autoencoder_run(
        mark_feat: list, mark_idx: list,
        frames_in: np.ndarray, frames_out: np.ndarray, frames_mean: np.ndarray,
        cluster_no: np.ndarray, take_frames: list,
        data_classname=None, data_labeled: bool=False,
        path2save: str='', show_plot: bool=False
) -> None:
    """"""
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 10})
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
        axs[1].scatter(mark_feat[mark_idx[0]][i], mark_feat[mark_idx[1]][i], color=get_plot_color(i), marker='.')
    axs[1].set_title('Feature Space')
    axs[1].set_ylabel(f'Feat[{mark_idx[0]}]')
    axs[1].set_xlabel(f'Feat[{mark_idx[1]}]')
    if isinstance(data_classname, list):
        if not len(data_classname) == 0:
            axs[1].legend(data_classname)

    # Denoised output
    if data_labeled:
        for i, id in enumerate(cluster_no):
            axs[2].plot(frames_mean[id, :], color=get_plot_color(i), linewidth=2)
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
    if show_plot:
        plt.show(block=True)
