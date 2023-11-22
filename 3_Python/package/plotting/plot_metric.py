import numpy as np
from os import mkdir
from os.path import exists, join
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from package.plotting.plot_common import save_figure


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
    plt.figure()
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


def plot_confusion(true_labels: list | np.ndarray, pred_labels: list | np.ndarray,
                   cl_dict=None, path2save="", title='Spike Sorting',
                   do_xticks_vertical=False) -> None:
    """Plotting the Confusion Matrix"""
    dict_available = isinstance(cl_dict, list)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.title(title)
    if dict_available:
        ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels,
            cmap=plt.cm.Blues, normalize='pred',
            colorbar=False, values_format='.2f',
            text_kw={'fontsize': 7},
            display_labels=cl_dict, xticks_rotation=('vertical' if do_xticks_vertical else 'horizontal')
        )
    else:
        ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels, y_pred=pred_labels,
            cmap=plt.cm.Blues, normalize='pred',
            colorbar=False, values_format='.2f',
            text_kw={'fontsize': 7}
        )

    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, f"confusion_matrix")


def _get_median(parameter: list) -> float:
    """Calculating the spectrum of the parameter"""
    param = np.zeros(shape=(len(parameter), ), dtype=float)
    for idx, val in enumerate(parameter):
        param[idx] = np.median(val)

    return float(np.median(param))
