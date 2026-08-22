import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_fscore_support

from denspp.offline.plot_helper import cm_to_inch, save_figure


def plot_confusion(
    true_labels: list | np.ndarray,
    pred_labels: list | np.ndarray,
    plotting: str = "class",
    show_accuracy: bool = False,
    cl_dict=None,
    path2save: str = "",
    name_addon: str = "",
    timestamps_result: list = (),
    timestamps_f1: list = (),
    timestamps_accuracy: list = (),
    show_plots: bool = False,
) -> None:
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
    cl_used = None
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

    precision, recall, fbeta, _ = precision_recall_fscore_support(
        true_labels, pred_labels, average="weighted"
    )

    do_xticks_vertical = False
    if plotting == "timestamps" or plotting == "both":
        # --- Plotting the results for the timestamp comparison
        plt.imshow(timestamps_result, cmap=plt.cm.Blues, interpolation="nearest")
        for i in range(timestamps_result.shape[0]):
            for j in range(timestamps_result.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{timestamps_result[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                )
        xtick_labels = ["true", "false"]
        plt.xticks(np.arange(2), xtick_labels)
        ytick_labels = ["positive", "negative"]
        plt.yticks(np.arange(2), ytick_labels)
        if show_accuracy:
            plt.title(f"F1-Score = {timestamps_f1:.4f} - Accuracy = {timestamps_accuracy:.4f}")
        else:
            plt.title(f"F1-Score = {timestamps_f1:.4f}")
    elif dict_available:
        use_cl_dict = list()
        for idx in np.unique(true_labels):
            use_cl_dict.append(cl_used[idx])
        do_xticks_vertical = np.unique(true_labels).size > 5 and len(use_cl_dict) > 5

        cmp = ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels,
            y_pred=pred_labels,
            normalize="pred",
            display_labels=use_cl_dict,
        )
    else:
        cmp = ConfusionMatrixDisplay.from_predictions(
            y_true=true_labels,
            y_pred=pred_labels,
            normalize="pred",
        )

    # --- Plotting the results of the class confusion matrix
    ax = plt.subplots(figsize=(cm_to_inch(12), cm_to_inch(12.5)))[1]
    cmp.plot(
        ax=ax,
        colorbar=False,
        values_format=".2f",
        text_kw={"fontsize": 6},
        cmap=plt.cm.Blues,
        xticks_rotation=(75 if do_xticks_vertical else "horizontal"),
    )
    ax.tick_params(axis="both", labelsize=10)
    cmp.ax_.set_title(f"Precision = {precision:.3f}, Recall = {recall:.3f}", fontsize=12)
    plt.tight_layout()
    # --- saving
    if path2save:
        save_figure(plt, path2save, f"confusion_matrix{name_addon}")
    if show_plots:
        plt.show(block=True)
