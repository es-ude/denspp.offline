import numpy as np
from os import mkdir
from os.path import join, exists
from scipy.io import loadmat
import matplotlib.pyplot as plt
from package.plot.plot_metric import plot_confusion


def logic_combination(true_labels: np.ndarray, pred_labels: np.ndarray, translate_dict: list) -> [np.ndarray,
                                                                                                  np.ndarray]:
    """Combination of logic"""
    true_labels_new = np.zeros(shape=true_labels.shape, dtype=np.uint8)
    pred_labels_new = np.zeros(shape=pred_labels.shape, dtype=np.uint8)

    for idx, cluster in enumerate(translate_dict):
        for id in cluster:
            pos = np.argwhere(true_labels == id).flatten()
            true_labels_new[pos] = idx
            pos = np.argwhere(pred_labels == id).flatten()
            pred_labels_new[pos] = idx
    return true_labels_new, pred_labels_new


if __name__ == '__main__':
    logsdir = '../runs/20231204_013929_train_rgc_class_v2/'
    data_result = loadmat(join(logsdir, 'results.mat'))
    path2save = join(logsdir, 'logic_comb')
    if not exists(path2save):
        mkdir(path2save)

    cell_dict_orig = data_result['cl_dict'].tolist()
    true_labels_orig = data_result['valid_clus'].flatten()
    pred_labels_orig = data_result['yclus'].flatten()

    # --- Logical combination for ON/OFF
    cell_dict_onoff = ['OFF', 'ON']
    translate_dict = [[0, 1], [2, 3]]
    true_labels_onoff, pred_labels_onoff = logic_combination(true_labels_orig, pred_labels_orig, translate_dict)
    plot_confusion(true_labels_onoff, pred_labels_onoff, 'training', cell_dict_onoff, path2save, '_logic_on-off')

    # --- Logical combination for ON/OFF
    cell_dict_transus = ['Sustained', 'Transient']
    translate_dict = [[0, 2], [1, 3]]
    true_labels_transus, pred_labels_transus = logic_combination(true_labels_orig, pred_labels_orig, translate_dict)
    plot_confusion(true_labels_transus, pred_labels_transus, 'training', cell_dict_transus, path2save, '_logic_transient-sustained')

    plt.show(block=True)
