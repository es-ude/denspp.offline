from os import makedirs
from os.path import join
from numpy import load

from package.data_call.call_cellbib import logic_combination
from package.dnn.plots.plot_metric import plot_confusion


def rgc_logic_combination(path2valid_data: str, valid_file_name='results_cl.npy', show_plot=False) -> None:
    """Post-Classification of Retinal Ganglion Celltype Classifier (RGC) after NN training
    Args:
        path2valid_data:    Path to validation data (generated after training)
        valid_file_name:    Filename of validation data [Default: 'results_cl.npy']
        show_plot:          Showing all plots
    Return:
        None
    """
    data_result = load(join(path2valid_data, valid_file_name), allow_pickle=True).item()
    path2save = join(path2valid_data, 'logic_comb')
    makedirs(path2save, exist_ok=True)

    cell_dict_orig = data_result['cl_dict']
    true_labels_orig = data_result['valid_clus']
    pred_labels_orig = data_result['yclus']

    # --- Logical combination for ON/OFF
    cell_dict_onoff = ['OFF', 'ON']
    translate_dict = [[0, 1], [2, 3]]
    true_labels_onoff, pred_labels_onoff = logic_combination(true_labels_orig, pred_labels_orig, translate_dict)
    plot_confusion(true_labels_onoff, pred_labels_onoff, 'class',
                   cl_dict=cell_dict_onoff, path2save=path2save,
                   name_addon='_logic_on-off')

    # --- Logical combination for ON/OFF
    cell_dict_transus = ['Sustained', 'Transient']
    translate_dict = [[0, 2], [1, 3]]
    true_labels_transus, pred_labels_transus = logic_combination(true_labels_orig, pred_labels_orig, translate_dict)
    plot_confusion(true_labels_transus, pred_labels_transus, 'class',
                   cl_dict=cell_dict_transus, path2save=path2save,
                   name_addon='_logic_transient-sustained', show_plots=show_plot)
