import numpy as np
from os import mkdir
from os.path import exists, join
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, precision_recall_fscore_support
from package.plot.plot_common import save_figure, cm_to_inch
from package.metric import compare_timestamps


def plot_boxplot_metric(freq: np.ndarray, metric: list, type_name: str, name: str,
                        path2save='', saving_formats=['pdf']) -> None:
    """Plotting one metric of the sweep run"""
    if not exists(path2save):
        mkdir(path2save)

    # --- Pre-Processing
    w = 0.1
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
    mean_metric = _get_median(metric)

    plt.clf()
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.serif': 'Times New Roman',
                         "lines.linewidth": 1, 'lines.marker': '.', 'lines.markersize': 12})

    plt.title(f"{type_name} = {mean_metric:.4f}")
    plt.boxplot(metric, positions=freq, widths=width(freq, w), patch_artist=True, showfliers=False)
    plt.xscale('log')
    plt.xlabel(r'Sampling Frequency f [Hz]')
    plt.ylabel(f"{type_name}")
    plt.grid()

    plt.tight_layout(pad=0.2)
    for type in saving_formats:
        plt.savefig(join(path2save, f"{name}_metric-box_{type_name}.{type}"), format=type)


def plot_loss(metric: list, metric_type: str, path2save='', epoch_zoom=None) -> None:
    """Plotting the loss of any DNN-based learning method"""
    # --- Pre-Processing
    plot_metrics = np.zeros(shape=(len(metric), 2), dtype=float)
    for idx, val in enumerate(metric):
        plot_metrics[idx, :] = np.array(val, dtype=float)

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(8)))
    axs = list()
    axs.append(plt.subplot(1, 1, 1))
    axs[0].plot(plot_metrics[:, 0], color='k', marker='.', label='Train.')
    axs[0].plot(plot_metrics[:, 1], color='r', marker='.', label='Valid.')
    plt.grid()
    plt.legend()
    plt.title(f"{metric_type} = {plot_metrics.max():.3f}")
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_type}')

    # --- Plot zooming component
    if isinstance(epoch_zoom, list) and len(epoch_zoom) > 0:
        x0 = int(epoch_zoom[0])
        x1 = int(epoch_zoom[1]) if len(epoch_zoom) == 2 else int(plot_metrics.shape[0]-1)
        pos = np.arange(x0, x1)
        min_value = np.min((plot_metrics[pos, 0], plot_metrics[pos, 1]))
        max_value = np.max((plot_metrics[pos, 0], plot_metrics[pos, 1]))

        axins0 = axs[0].inset_axes(
            [0.45, 0.02, 0.5, 0.43],
            xticklabels=[],
            # yticklabels=[],
            xlim=(x0-0.5, x1+0.5), ylim=(0.99 * min_value, 1.01 * max_value))
        axins0.plot(plot_metrics[:, 0], color='k', marker='.', label='Train.')
        axins0.plot(plot_metrics[:, 1], color='r', marker='.', label='Valid.')
        axins0.grid()
        axs[0].tick_params(direction='in')
        axs[0].indicate_inset_zoom(axins0, edgecolor="black")
        addon = '_zoomed'
    else:
        addon = ''

    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"loss_metric_{metric_type}"+addon)


def plot_confusion(true_labels: list | np.ndarray,
                    pred_labels: list | np.ndarray,
                    timestamps_result,
                    timestamps_f1, timestamps_accuracy,
                    plotting="class",
                    show_accuracy=False,
                    cl_dict=None,
                    path2save="", name_addon=""
                    ) -> None:
    """This function is designed to generate and display confusion matrices for classification results and
    timestamp-based comparisons. The function takes various parameters, including true and predicted labels, as well as
    additional information such as timestamps and plotting preferences. The confusion matrix for classification is
    displayed using the ConfusionMatrixDisplay class from scikit-learn. It also calculates and prints precision, recall,
     and F-beta score for the classification case. The timestamp-based comparison is visualized using a heatmap.
     The resulting plots can be saved to a specified path if provided.

    Parameters:
        true_labels: List or numpy array containing true class labels.
        pred_labels: List or numpy array containing predicted class labels.
        timestamps_result: Resulting array for timestamp comparison (for plotting timestamps).
        timestamps_f1: F1-score for timestamp comparison.
        timestamps_accuracy: Accuracy for timestamp comparison.
        plotting: Specifies the type of plotting to perform ("class", "timestamps", or "both").
        show_accuracy: Boolean indicating whether to display accuracy in timestamp plots.
        cl_dict: Dictionary mapping class indices to labels.
        path2save: Path to save the generated plots.
        name_addon: Additional name for saved plots.

    Output:
        The function generates and displays confusion matrices and timestamp-based plots.
        If path2save is provided, the plots are saved to the specified path."""

    if plotting == "class" or plotting == "both":
        """Plotting the Confusion Matrix"""
        if isinstance(cl_dict, np.ndarray):
            cl_used = cl_dict.tolist()
        else:
            cl_used = cl_dict

        if isinstance(cl_dict, list):
            dict_available = not len(cl_dict) == 0
        else:
            dict_available = False

        max_key_length = 0

        precision, recall, fbeta, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
        if dict_available:
            for keys in cl_used:
                max_key_length = len(keys) if len(keys) > max_key_length else max_key_length
            do_xticks_vertical = bool(max_key_length > 5) and np.unique(true_labels).size > 3
            use_cl_dict = list()
            for idx in np.unique(true_labels):
                use_cl_dict.append(cl_used[int(idx)])

            cmp = ConfusionMatrixDisplay.from_predictions(
                y_true=true_labels, y_pred=pred_labels, normalize='pred', display_labels=use_cl_dict
            )
        else:
            do_xticks_vertical = False
            cmp = ConfusionMatrixDisplay.from_predictions(
                y_true=true_labels, y_pred=pred_labels, normalize='pred'
            )

        # --- Plotting the results of the class confusion matrix
        fig, ax = plt.subplots(figsize=(cm_to_inch(12), cm_to_inch(12.5)))
        cmp.plot(ax=ax, colorbar=False, values_format='.3f',
                 text_kw={'fontsize': 9}, cmap=plt.cm.Blues,
                 xticks_rotation=('vertical' if do_xticks_vertical else 'horizontal')
                 )
        cmp.ax_.set_title(f'Precision = {100*precision:.2f}%, Recall = {100*recall:.2f}%')
        print(f'... Fbeta score is {100*fbeta:.2f}%')
        plt.tight_layout()
        if path2save:
            save_figure(plt, path2save, f"confusion_matrix_classes{name_addon}")
        plt.close('all')

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
        plt.tight_layout()
        print(path2save)
        if path2save:
            save_figure(plt, path2save, f"confusion_matrix_timestamps")
        plt.close('all')


def prep_confusion(true_labels: list, pred_labels: list, mode="training", plots="class", show_accuracy=False,
                   cl_dict=None, path2save="", window=2):

    """This function serves as a wrapper for the _plot_confusion function, primarily focused on preparing and organizing
     the inputs for the visualization of confusion matrices. It supports two modes: "pipeline" and other modes.
     In "pipeline" mode, it computes true positive (TP), false positive (FP), false negative (FN), F1-score, and
     accuracy using the compare_timestamps function. It then generates a timestamp-based result matrix and calls
     _plot_confusion for visualization. In other modes, it directly calls _plot_confusion with classification plotting.

    Parameters:
        true_labels: List or numpy array containing true class labels.
        pred_labels: List or numpy array containing predicted class labels.
        mode: Specifies the mode ("pipeline" or other) for different processing.
        plots: Specifies the type of plotting to perform ("class", "timestamps", or "both").
        show_accuracy: Boolean indicating whether to display accuracy in timestamp plots.
        cl_dict: Dictionary mapping class indices to labels.
        path2save: Path to save the generated plots.
        window: Window parameter for timestamp-based comparisons.

    Output:
        The function calls _plot_confusion with the appropriate parameters based on the specified mode, resulting in
        the visualization of confusion matrices. If path2save is provided, the plots are saved to the specified path."""

    if mode == "pipeline":
        TP, FP, FN, f1_score, accuracy, true_labels, pred_labels = compare_timestamps(true_labels, pred_labels, window)
        result = np.array([[TP, FP], [0, FN]])
        plot_confusion(true_labels, pred_labels, result, f1_score, accuracy, plots, show_accuracy, cl_dict, path2save)
    else:
        plot_confusion(true_labels, pred_labels, None, None, None, "class", False, cl_dict, path2save)


def _get_median(parameter: list) -> float:
    """Calculating the spectrum of the parameter"""
    param = np.zeros(shape=(len(parameter),), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.median(val)

    return float(np.median(param))
