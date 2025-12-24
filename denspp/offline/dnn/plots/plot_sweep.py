import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.plot_helper import (
    save_figure,
    get_plot_color
)


def plot_common_loss(metric: dict, path2save: str='', show_plots: bool=False) -> None:
    """Function for plotting the loss function of both models with sweeping the feature space size
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['loss_train', 'loss_valid']
    keys_cl = ['train_acc', 'valid_acc']

    ## --- Subplot #1: Loss / Acc.
    _, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ln0 = ax1.plot(feat_size, metric['ae'][keys_ae[0]], 'k.-', label='Loss_AE (Train)')
    ln1 = ax1.plot(feat_size, metric['ae'][keys_ae[1]], 'k.--', label='Loss_AE (Valid)')
    ax1.set_ylabel("Loss, Autoencoder", fontsize=14)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(feat_size, metric['cl'][keys_cl[0]], 'r.-', label='Acc._CL (Train)')
    ln3 = ax2.plot(feat_size, metric['cl'][keys_cl[1]], 'r.--', label='Acc._CL (Valid)')
    ax2.set_ylabel("Accuracy, Classifier", fontsize=14)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 7))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

    lns = ln0 + ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ## --- End processing
    ax1.grid()
    ax1.set_xlabel('Feature Size', fontsize=14)
    ax1.set_xticks(feat_size_ticks)
    ax1.set_xlim([feat_size[0], feat_size[-1]])
    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_common_loss', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_common_params(metric: dict, path2save: str= '', show_plots: bool=False) -> None:
    """Function for plotting the parameter numbers of both models with sweeping the feature space size
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['model_params']
    keys_cl = ['model_params']

    _, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ln4 = ax1.plot(feat_size, metric['ae'][keys_ae[0]], 'k.-', label='Autoencoder (AE)')
    ax2 = ax1.twinx()
    ln5 = ax2.plot(feat_size, metric['cl'][keys_cl[0]], 'r.-', label='Classifier (CL)')
    ax1.set_ylabel('Model parameters', fontsize=14)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 7))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

    lns = ln4 + ln5
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ## --- End processing
    ax1.grid()
    ax1.set_xlabel('Feature Size', fontsize=14)
    ax1.set_xticks(feat_size_ticks)
    ax1.set_xlim([feat_size[0], feat_size[-1]])
    plt.tight_layout()

    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_common_params', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_architecture_violin(metric: dict, path2save: str = '', show_plots: bool=False, label_dict=None) -> None:
    """Function for plotting the architecture violin plot
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :param label_dict:  Dictionary with Labels to extract feature size from
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['dsnr_cl']
    keys_cl = ['precision']

    num_cluster = metric['cl']['output_size'][-1]
    if label_dict is None:
        label_dict = [f'Neuron #{idx}' for idx in range(num_cluster)]

    for key0, key1 in zip(keys_ae, keys_cl):
        _, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
        data_ae = metric['ae'][key0]
        data_cl = metric['cl'][key1]

        # --- Processing AE Data
        transformed_data_ae_cluster = [[] for idx in range(num_cluster)]
        for data_feat in data_ae:
            for idx, data_cluster in enumerate(data_feat):
                transformed_data_ae_cluster[idx].append(data_cluster)

        for idx, data_boxplot in enumerate(transformed_data_ae_cluster):
            axs[0].violinplot(data_boxplot, showmedians=True, positions=np.array(feat_size)+idx/num_cluster, widths=0.5/num_cluster)
        axs[0].legend(label_dict)
        axs[0].set_ylabel(key0, fontsize=14)

        # --- Processing CL Data
        transformed_data_cl_cluster = np.zeros((num_cluster, len(feat_size)))
        for idx, data_cluster in enumerate(data_cl):
            transformed_data_cl_cluster[:, idx] = data_cluster
        for idx, data_plot in enumerate(transformed_data_cl_cluster):
            axs[1].plot(feat_size, data_plot, f'{get_plot_color(idx)}.-', label=label_dict[idx])
        axs[1].set_ylabel(key1, fontsize=14)
        axs[1].legend()

        ## --- End processing
        for ax in axs:
            ax.grid()
            ax.set_xlabel('Feature Size', fontsize=14)
        axs[0].set_xticks(feat_size_ticks)
        axs[0].set_xlim([feat_size[0]-1, feat_size[-1]+1])
        plt.tight_layout()

    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_architecture_metrics_isolated(metric: dict, path2save: str = '', show_plots: bool=False, label_dict=None) -> None:
    """Function for plotting the metrics in isolated plots for the autoencoder and classifier model
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :param label_dict:  Dictionary with Labels to extract feature size from
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)

    keys_ae = ['dsnr_cl']
    keys_cl = ['precision']

    num_cluster = metric['cl']['output_size'][-1]
    if label_dict is None:
        label_dict = [f'Neuron #{idx}' for idx in range(num_cluster)]

    num_rows = 2
    num_cols = 3

    # --- Figure #1: Autoencoder
    for key0 in keys_ae:
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True)
        data_ae = metric['ae'][key0]
        transformed_data_ae_cluster = [[] for idx in range(num_cluster)]
        transformed_data_ae_median = np.zeros((num_cluster, len(feat_size)))
        for idy, data_feat in enumerate(data_ae):
            for idx, data_cluster in enumerate(data_feat):
                transformed_data_ae_cluster[idx].append(data_cluster)
                transformed_data_ae_median[idx, idy] = np.median(data_cluster)
                pass

        for idx, data_boxplot in enumerate(transformed_data_ae_cluster):
            axs[int(idx / num_cols), idx % num_cols].plot(feat_size, transformed_data_ae_median[idx, :], 'k.--', linewidth=1.0)
            axs[int(idx/num_cols), idx % num_cols].violinplot(data_boxplot, showmedians=True, positions=feat_size)
            axs[int(idx/num_cols), idx % num_cols].set_ylabel(f'{key0} ({label_dict[idx]})', fontsize=14)
            axs[int(idx/num_cols), idx % num_cols].grid()

        ## --- End processing
        axs[1, 1].set_xlabel('Feature Size', fontsize=14)
        axs[0, 0].set_xticks(feat_size_ticks)
        axs[0, 0].set_xlim([feat_size[0] - 0.25, feat_size[-1] + 0.25])

    plt.subplots_adjust(wspace=0.3, hspace=0.05)
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture_ae', ['svg'])

    # --- Figure #2: Classifier
    for key1 in keys_cl:
        _, axs = plt.subplots(nrows=1, ncols=len(keys_cl), sharex=True)
        # --- Processing CL Data
        data_cl = metric['cl'][key1]
        transformed_data_cl_cluster = np.zeros((num_cluster, len(feat_size)))
        for idx, data_cluster in enumerate(data_cl):
            transformed_data_cl_cluster[:, idx] = data_cluster
        for idx, data_plot in enumerate(transformed_data_cl_cluster):
            axs.plot(feat_size, data_plot, f'{get_plot_color(idx)}.-', label=label_dict[idx])
        axs.set_ylabel(key1, fontsize=14)
        axs.legend()
        axs.grid()

        ## --- End processing
        axs.set_xlabel('Feature Size', fontsize=14)
        axs.set_xticks(feat_size_ticks)
        axs.set_xlim([feat_size[0]-0.25, feat_size[-1]+0.25])

    plt.subplots_adjust(wspace=0.3, hspace=0.05)
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture_cl', ['svg'])
    if show_plots:
        plt.show(block=True)
