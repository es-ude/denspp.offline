import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

from package.plot_helper import save_figure, cm_to_inch
from package.metric.timestamps import compare_timestamps


def _get_median(parameter: list) -> float:
    """Calculating the spectrum of the parameter"""
    param = np.zeros(shape=(len(parameter),), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.median(val)

    return float(np.median(param))


def plot_loss(loss_train: list, loss_valid: list, type: str, path2save='', epoch_zoom=None, show_plot=False) -> None:
    """Plotting the loss of any DNN-based learning method
    Args:
        loss_train:     List with loss values from training
        loss_valid:     List with loss values from validation
        type:           Name of the metric type
        path2save:      Path to save the figure
        epoch_zoom:     Do zoom on defined range [list]
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
    plt.xticks(pos)
    plt.xlim([pos[0], pos[-1]])

    plt.grid()
    plt.legend()
    plt.title(f"{type} = {plot_metrics.max() if 'Acc' in type else plot_metrics.min():.3f}")
    plt.xlabel('Epoch')
    plt.ylabel(f'{type}')

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
        save_figure(plt, path2save, f"loss_metric_{type}" + addon)
    if show_plot:
        plt.show(block=True)


def plot_confusion(true_labels: list | np.ndarray,
                   pred_labels: list | np.ndarray,
                   plotting="class",
                   show_accuracy=False,
                   cl_dict=None,
                   path2save="", name_addon="",
                   timestamps_result=(),
                   timestamps_f1=(), timestamps_accuracy=(),
                   show_plots=False) -> None:
    """This function is designed to generate and display confusion matrices for classification results and
    timestamp-based comparisons. The function takes various parameters, including true and predicted labels, as well as additional information such as timestamps and plotting preferences. The confusion matrix for classification is
    displayed using the ConfusionMatrixDisplay class from scikit-learn. It also calculates and prints precision, recall,
    and F-beta score for the classification case. The timestamp-based comparison is visualized using a heatmap.
    The resulting plots can be saved to a specified path if provided.
    Args:
        true_labels:            List or numpy array containing true class labels.
        pred_labels:            List or numpy array containing predicted class labels.
        timestamps_result:      Resulting array for timestamp comparison (for plotting timestamps).
        timestamps_f1:          F1-score for timestamp comparison.
        timestamps_accuracy:    Accuracy for timestamp comparison.
        plotting:               Specifies the type of plotting to perform ("class", "timestamps", or "both").
        show_accuracy:          Boolean indicating whether to display accuracy in timestamp plots.
        cl_dict:                Dictionary mapping class indices to labels.
        path2save:              Path to save the generated plots.
        name_addon:             Additional name for saved plots.
        show_plots:             Command for showing plots in the end [Default: False]
    Returns:
        The function generates and displays confusion matrices and timestamp-based plots.
        If path2save is provided, the plots are saved to the specified path.
    """
    dict_available = False
    if plotting == "class" or plotting == "both":
        """Plotting the Confusion Matrix"""
        if isinstance(cl_dict, np.ndarray):
            cl_used = cl_dict.tolist()
        else:
            cl_used = cl_dict

        if isinstance(cl_dict, list):
            dict_available = not len(cl_dict) == 0
        elif isinstance(cl_dict, dict):
            dict_available = not len(cl_dict) == 0
        else:
            dict_available = False

    max_key_length = 0
    precision, recall, fbeta, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    if plotting == "timestamps" or plotting == "both":
        # --- Plotting the results for the timestamp comparison
        plt.imshow(timestamps_result, cmap=plt.cm.Blues, interpolation='nearest')
        for i in range(timestamps_result.shape[0]):
            for j in range(timestamps_result.shape[1]):
                plt.text(j, i, f'{timestamps_result[i, j]:.2f}', ha='center', va='center', color='white')
        xtick_labels = ['true', 'false']
        plt.xticks(np.arange(2), xtick_labels)
        ytick_labels = ['positive', 'negative']
        plt.yticks(np.arange(2), ytick_labels)
        if show_accuracy:
            plt.title(f'F1-Score = {timestamps_f1:.4f} - Accuracy = {timestamps_accuracy:.4f}')
        else:
            plt.title(f'F1-Score = {timestamps_f1:.4f}')
    elif dict_available:
        for keys in cl_used:
            max_key_length = len(keys) if len(keys) > max_key_length else max_key_length
        do_xticks_vertical = bool(max_key_length > 5) and np.unique(true_labels).size > 3
        use_cl_dict = list()
        if isinstance(cl_dict, dict):
            for key in cl_dict.keys():
                use_cl_dict.append(key)
        else:
            for idx in np.unique(true_labels):
                use_cl_dict.append(cl_used[int(idx)])

        cmp = ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels, normalize='pred', display_labels=use_cl_dict
        )
    else:
        do_xticks_vertical = False
        cmp = ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels, normalize='pred',
        )

    # --- Plotting the results of the class confusion matrix
    ax = plt.subplots(figsize=(cm_to_inch(12), cm_to_inch(12.5)))[1]
    cmp.plot(ax=ax, colorbar=False, values_format='.3f',
             text_kw={'fontsize': 8}, cmap=plt.cm.Blues,
             xticks_rotation=('vertical' if do_xticks_vertical else 'horizontal'))
    cmp.ax_.set_title(f'Precision = {100*precision:.2f}%, Recall = {100*recall:.2f}%')
    plt.tight_layout()
    # --- saving
    if path2save:
        save_figure(plt, path2save, f"confusion_matrix{name_addon}")
    if show_plots:
        plt.show(block=True)


def prep_confusion(true_labels: list, pred_labels: list, mode="training", plots="class", show_accuracy=False,
                   cl_dict=None, path2save="", window=2) -> None:
    """This function serves as a wrapper for the _plot_confusion function, primarily focused on preparing and organizing
     the inputs for the visualization of confusion matrices. It supports two modes: "pipeline" and other modes.
     In "pipeline" mode, it computes true positive (TP), false positive (FP), false negative (FN), F1-score, and
     accuracy using the compare_timestamps function. It then generates a timestamp-based result matrix and calls
     _plot_confusion for visualization. In other modes, it directly calls _plot_confusion with classification plotting.

    Args:
        true_labels: List or numpy array containing true class labels.
        pred_labels: List or numpy array containing predicted class labels.
        mode: Specifies the mode ("pipeline" or other) for different processing.
        plots: Specifies the type of plotting to perform ("class", "timestamps", or "both").
        show_accuracy: Boolean indicating whether to display accuracy in timestamp plots.
        cl_dict: Dictionary mapping class indices to labels.
        path2save: Path to save the generated plots.
        window: Window parameter for timestamp-based comparisons.
    Returns:
        The function calls _plot_confusion with the appropriate parameters based on the specified mode, resulting in
        the visualization of confusion matrices. If path2save is provided, the plots are saved to the specified path."""

    if mode == "pipeline":
        TP, FP, FN, f1_score, accuracy, true_labels, pred_labels = compare_timestamps(true_labels, pred_labels, window)
        result = np.array([[TP, FP], [0, FN]])
        plot_confusion(true_labels, pred_labels, result, f1_score, accuracy, plots, show_accuracy, cl_dict, path2save)
    else:
        plot_confusion(true_labels, pred_labels, None, None, None, "class", False, cl_dict, path2save)


def plot_statistic_data(train_cl: np.ndarray | list, valid_cl=None, path2save='',
                        cl_dict=None, show_plot=False) -> None:
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
        save_figure(plt, path2save, "ai_training_histdata")
    if show_plot:
        plt.show(block=True)
