import numpy as np
from matplotlib import pyplot as plt

from denspp.offline.plot_helper import (
    save_figure,
    cm_to_inch,
    get_plot_color,
    get_textsize_paper
)


def plot_loss(loss_train: list, loss_valid: list, loss_type: str, path2save: str= '',
              epoch_zoom=None, do_logy: bool=False, show_plot: bool=False) -> None:
    """Plotting the loss of any DNN-based learning method
    Args:
        loss_train:     List with loss values from training
        loss_valid:     List with loss values from validation
        loss_type:      Name of the metric type
        path2save:      Path to save the figure
        epoch_zoom:     Do zoom on defined range [list]
        do_logy:        Boolean for logarithmic y-axis scale
        show_plot:      Showing and blocking the plots
    Return:
        None
    """
    # --- Pre-Processing
    plot_metrics = np.zeros((len(loss_train), 2), dtype=float)
    plot_metrics[:, 0] = np.array(loss_train)
    plot_metrics[:, 1] = np.array(loss_valid)

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(8)))
    axs = list()
    axs.append(plt.subplot(1, 1, 1))

    epochs_ite = np.array([idx + 1 for idx in range(plot_metrics[:, 0].size)])
    axs[0].plot(epochs_ite, plot_metrics[:, 0], color='k', marker='.', label='Train.')
    axs[0].plot(epochs_ite, plot_metrics[:, 1], color='r', marker='.', label='Valid.')

    pos = np.linspace(epochs_ite[0], epochs_ite[-1], num=11, endpoint=True, dtype=int)
    plt.yscale('log' if do_logy else 'linear')
    plt.xticks(pos)
    plt.xlim([pos[0], pos[-1]])

    plt.grid()
    plt.legend()
    plt.title(f"{loss_type} = {plot_metrics.max() if 'Acc' in loss_type else plot_metrics.min():.3f}")
    plt.xlabel('Epoch')
    plt.ylabel(f'{loss_type}')

    # --- Plot zooming component
    if isinstance(epoch_zoom, list) and len(epoch_zoom) > 0:
        x0 = int(epoch_zoom[0])
        x1 = int(epoch_zoom[1]) if len(epoch_zoom) == 2 else int(plot_metrics.shape[0]-1)
        pos = np.arange(x0, x1)
        min_value = np.min((plot_metrics[pos, 0], plot_metrics[pos, 1]))
        max_value = np.max((plot_metrics[pos, 0], plot_metrics[pos, 1]))

        axins0 = axs[0].inset_axes([0.45, 0.02, 0.5, 0.43], xticklabels=[],
                                   xlim=(x0-0.5, x1+0.5), ylim=(0.99 * min_value, 1.01 * max_value))
        axins0.plot(epochs_ite, plot_metrics[:, 0], color='k', marker='.', label='Train.')
        axins0.plot(epochs_ite, plot_metrics[:, 1], color='r', marker='.', label='Valid.')
        axins0.grid()
        axs[0].tick_params(direction='in')
        axs[0].indicate_inset_zoom(axins0, edgecolor="black")
        addon = '_zoomed'
    else:
        addon = ''

    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"report_metric_{loss_type}" + addon)
    if show_plot:
        plt.show(block=True)


def plot_custom_loss_classifier(data: list, loss_name: str, fold_num: int, do_logy: bool=False, epoch_zoom=None, path2save: str= '', show_plot: bool=False) -> None:
    """Plotting the custom loss of each epoch from training (e.g. Signal-to-Noise Ratio (SNR)
    :param data:        List with values from each epoch
    :param loss_name:   String with name of loss function
    :param fold_num:    Integer with fold number
    :param do_logy:     Boolean for plotting the y-axis logarithmic scale
    :param epoch_zoom:  Do zoom on defined range [list]
    :param path2save:   Path to save the figure
    :param show_plot:   If true, show plot
    :return:            None
    """
    epochs_ite = np.array([idx + 1 for idx in range(len(data))])
    metric0 = np.array(data) if not 'ptq_loss' in loss_name else np.array(data).flatten()
    pos = np.linspace(epochs_ite[0], epochs_ite[-1], num=11, endpoint=True, dtype=int)

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': get_textsize_paper()})
    axs = list()
    axs.append(plt.subplot(1, 1, 1))
    axs[0].plot(epochs_ite, metric0, marker='.', markersize=5)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel(f'{loss_name}')
    axs[0].set_yscale('log' if do_logy else 'linear')
    # --- Plot zooming component
    if isinstance(epoch_zoom, list) and len(epoch_zoom) > 0:
        x0 = int(epoch_zoom[0])
        x1 = int(epoch_zoom[1]) if len(epoch_zoom) == 2 else int(metric0.shape[0] - 1)
        pos = np.arange(x0, x1)
        min_value = np.min(metric0[pos, :])
        max_value = np.max(metric0[pos, :])

        axins0 = axs[0].inset_axes([0.45, 0.02, 0.5, 0.43], xticklabels=[],
                                   xlim=(x0 - 0.5, x1 + 0.5), ylim=(0.99 * min_value, 1.01 * max_value))
        axins0.plot(epochs_ite, metric0, color='k', marker='.', markersize=6)
        axins0.grid()
        axs[0].tick_params(direction='in')
        axs[0].indicate_inset_zoom(axins0, edgecolor="black")
        addon = '_zoomed'
    else:
        addon = ''

    plt.xticks(pos)
    plt.xlim([pos[0], pos[-1]])
    plt.grid()
    plt.tight_layout(pad=0.5)
    if path2save:
        save_figure(plt, path2save, f"report_metric_{loss_name}_fold{fold_num:03d}" + addon)
    if show_plot:
        plt.show(block=True)


def plot_custom_loss_autoencoder(data: list, loss_name: str, do_boxplot: bool=False, do_logy: bool=False, epoch_zoom=None, path2save: str= '', show_plot: bool=False) -> None:
    """Plotting the custom loss of each epoch from training (e.g. Signal-to-Noise Ratio (SNR)
    :param data:        List with values from each epoch
    :param loss_name:   String with name of loss function
    :param do_boxplot:  If true, plot boxplot
    :param do_logy:     Boolean for plotting the y-axis logarithmic scale
    :param epoch_zoom:  Do zoom on defined range [list]
    :param path2save:   Path to save the figure
    :param show_plot:   If true, show plot
    :return:            None
    """
    metric_processed = list()
    if not do_boxplot:
        metric0 = np.zeros(shape=(len(data), 3), dtype=float)
        for idx, metric_epoch in enumerate(data):
            metric0[idx, :] = metric_epoch.min(), np.median(metric_epoch), metric_epoch.max()
        metric_processed.append(metric0)
    else:
        metric_processed.append(data)

    # --- Plotting
    for idx, metric0 in enumerate(metric_processed):
        epochs_ite = np.array([idx + 1 for idx in range(metric0[:, 0].size)])
        pos = np.linspace(epochs_ite[0], epochs_ite[-1], num=11, endpoint=True, dtype=int)

        plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
        plt.rcParams.update({'font.size': get_textsize_paper()})
        axs = list()
        axs.append(plt.subplot(1, 1, 1))
        axs[0].plot(epochs_ite, metric0, marker='.', markersize=5)
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel(f'{loss_name} (dB)')
        axs[0].set_yscale('log' if do_logy else 'linear')

        if not do_boxplot:
            plt.plot(epochs_ite,metric0[:, 0], color=get_plot_color(0), marker='.', label='min')
            plt.plot(epochs_ite, metric0[:, 1], color=get_plot_color(1), marker='.', label='mean')
            plt.plot(epochs_ite, metric0[:, 2], color=get_plot_color(2), marker='.', label='max')
            plt.legend()
        else:
            plt.boxplot(data, patch_artist=True, showfliers=False)

        # --- Plot zooming component
        if isinstance(epoch_zoom, list) and len(epoch_zoom) > 0:
            x0 = int(epoch_zoom[0])
            x1 = int(epoch_zoom[1]) if len(epoch_zoom) == 2 else int(metric0.shape[0] - 1)
            pos = np.arange(x0, x1)
            min_value = np.min(metric0[pos, :])
            max_value = np.max(metric0[pos, :])

            axins0 = axs[0].inset_axes([0.45, 0.02, 0.5, 0.43], xticklabels=[],
                                       xlim=(x0 - 0.5, x1 + 0.5), ylim=(0.99 * min_value, 1.01 * max_value))
            axins0.plot(epochs_ite, metric0, color='k', marker='.', markersize=6)
            axins0.grid()
            axs[0].tick_params(direction='in')
            axs[0].indicate_inset_zoom(axins0, edgecolor="black")
            addon = '_zoomed'
        else:
            addon = ''

        plt.xlim([pos[0], pos[-1]])
        plt.grid()
        plt.tight_layout(pad=0.5)
        if path2save:
            save_figure(plt, path2save, f"report_metric_{loss_name}_fold{idx:03d}" + addon)
        if show_plot:
            plt.show(block=True)


def plot_statistic(train_cl: np.ndarray, valid_cl=None, path2save: str= '',
                   cl_dict=None, show_plot: bool=False) -> None:
    """Plotting the statistics of used dataset during training
    :param train_cl:    Numpy array of all classification labels from training dataset
    :param valid_cl:    Numpy array of all classification labels from validation dataset (optional)
    :param path2save:   Path to save the figure
    :param cl_dict:     Dictionary with keys for each label (otherwise number are used)
    :param show_plot:   If true, show plot
    :return:            None
    """
    do_plots_avai = isinstance(valid_cl, np.ndarray | list)
    dict_available = isinstance(cl_dict, np.ndarray | list | dict)
    use_cl_dict = list()
    if dict_available:
        if isinstance(cl_dict, np.ndarray):
            cl_dict0 = cl_dict.tolist()
        else:
            cl_dict0 = cl_dict
        xtick_text = 'vertical' if len(cl_dict0) > 3 else 'horizontal'
    else:
        xtick_text = 'horizontal'

    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(8)))
    plt.rcParams.update({'font.size': 12})
    plt.subplots_adjust(hspace=0, wspace=0.5)
    axs = list()
    for idx in range(0, 1+do_plots_avai):
        axs.append(plt.subplot(1, 1 + do_plots_avai, 1+idx))

    # Histogram of Training data
    check = np.unique(train_cl, return_counts=True)
    axs[0].bar(check[0], check[1], color='k', width=0.8)
    if dict_available:
        if not len(cl_dict) == 0:
            if isinstance(cl_dict, dict):
                for key in cl_dict.keys():
                    use_cl_dict.append(key)
            else:
                for idx in np.unique(train_cl):
                    use_cl_dict.append(cl_dict[int(idx)])
            axs[0].set_xticks(check[0], (use_cl_dict if check[0].size != 1 else [use_cl_dict[0]]),
                              rotation=xtick_text)
    else:
        axs[0].set_xticks(check[0])

    axs[0].set_ylabel("Bins")
    axs[0].set_ylim([int(0.99*check[1].min()), int(1.01*check[1].max())])
    axs[0].set_title('Training')

    # Histogram of Validation data
    if do_plots_avai:
        check = np.unique(valid_cl, return_counts=True)
        axs[1].bar(check[0], check[1], color='k', width=0.8)
        if dict_available:
            if not len(cl_dict) == 0:
                axs[1].set_xticks(check[0], (use_cl_dict if check[0].size != 1 else [use_cl_dict[0]]),
                                  rotation=xtick_text)
        else:
            axs[0].set_xticks(check[0])

        axs[1].set_ylim([int(0.99 * check[1].min()), int(1.01 * check[1].max())])
        axs[1].set_title('Validation')

    for ax in axs:
        ax.grid()
        ax.set_xlabel("Cluster")

    plt.tight_layout(pad=0.5)
    # --- saving plots
    if path2save:
        save_figure(plt, path2save, "report_statistic_data")
    if show_plot:
        plt.show(block=True)
