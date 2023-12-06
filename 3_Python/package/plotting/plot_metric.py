import numpy as np
from os import mkdir
from os.path import exists, join
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, precision_recall_fscore_support
from package.plotting.plot_common import save_figure, cm_to_inch
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


def plot_loss(metric: list, metric_type: str, name='', path2save='') -> None:
    """Plotting the loss of any DNN-based learning method"""
    # --- Pre-Processing
    plot_metrics = np.zeros(shape=(len(metric), 2), dtype=float)
    for idx, val in enumerate(metric):
        plot_metrics[idx, :] = np.array(val, dtype=float)

    # --- Plotting
    plt.figure(figsize=(cm_to_inch(10), cm_to_inch(8)))
    plt.title(f"{metric_type} = {plot_metrics.max():.3f}")
    plt.plot(plot_metrics[:, 0], color='k', marker='.', label='Train.')
    plt.plot(plot_metrics[:, 1], color='r', marker='.', label='Valid.')
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric_type}')

    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"loss_metric_{metric_type}")


def plot_confusion_classes(true_labels: list | np.ndarray,
                   pred_labels: list | np.ndarray,
                   mode="training",
                   cl_dict=None, path2save="", window=2) -> None:
    if mode == "pipeline":
        _, _, _, _,_, true_labels, pred_labels = compare_timestamps(true_labels, pred_labels, window)
    """Plotting the Confusion Matrix"""
    dict_available = isinstance(cl_dict, list)
    max_key_length = 0

    if dict_available:
        for keys in cl_dict:
            max_key_length = len(keys) if len(keys) > max_key_length else max_key_length
        do_xticks_vertical = bool(max_key_length > 5) and np.unique(true_labels).size > 3
        use_cl_dict = list()
        for idx in np.unique(true_labels):
            use_cl_dict.append(cl_dict[int(idx)])

        ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels,
            cmap=plt.cm.Blues, normalize='pred',
            colorbar=False, values_format='.3f',
            text_kw={'fontsize': 7},
            display_labels=use_cl_dict,
            xticks_rotation=('vertical' if do_xticks_vertical else 'horizontal')
        )
    else:
        ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels,
            cmap=plt.cm.Blues, normalize='pred',
            colorbar=False, values_format='.3f',
            text_kw={'fontsize': 7}
        )
    # Determining Recall and Precision
    precision, recall, fbeta, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    plt.title(f'Precision = {precision:.4f} - Recall = {recall:.4f} - Fbeta = {fbeta:.4f}')
    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"confusion_matrix_classes")


def plot_confusion_timestamps(true_labels: list, pred_labels: list, show_accuracy=False, path2save="", window=2):
    TP, FP, FN, f1_score, accuracy, _, _ = compare_timestamps(true_labels, pred_labels, window)
    result = np.array([[TP, FP], [0, FN]])

    plt.imshow(result, cmap=plt.cm.Blues, interpolation='nearest')
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            plt.text(j, i, f'{result[i, j]:.2f}', ha='center', va='center', color='white')
    xtick_labels = ['true', 'false']
    plt.xticks(np.arange(2), xtick_labels)
    ytick_labels = ['positive', 'negative']
    plt.yticks(np.arange(2), ytick_labels)
    if show_accuracy:
        plt.title(f'F1-Score = {f1_score:.4f} - Accuracy = {accuracy:.4f}')
    else:
        plt.title(f'F1-Score = {f1_score:.4f}')
    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"confusion_matrix_timestamps")


def _get_median(parameter: list) -> float:
    """Calculating the spectrum of the parameter"""
    param = np.zeros(shape=(len(parameter), ), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.median(val)

    return float(np.median(param))
