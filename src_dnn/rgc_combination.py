from os import makedirs
from os.path import join
from numpy import load

from denspp.offline.data_call.call_cellbib import CellMergeClass, logic_combination
from denspp.offline.dnn.plots.plot_metric import plot_confusion


RetinalGanglionCellTDB = CellMergeClass(
    cell_type_to_id = {
        "ON-OFF DS - dorsal": 0,
        "ON-OFF DS - temporal": 1,
        "ON-OFF DS - ventral": 2,
        "ON-OFF DS - nasal": 3,
        "ON-OFF DS - direction unknown": 4,
        "ON-OFF - subtype unknown": 5,
        "ON DS sustained - dorsonasal": 6,
        "ON DS sustained - temporal": 7,
        "ON DS sustained - ventral": 8,
        "ON DS sustained - direction unknown": 9,
        "OFF sustained EW3o": 10,
        "OFF medium sustained": 11,
        "OFF vertical OS - asymmetric": 12,
        "OFF horizontal OS - symmetric": 13,
        "OFF sustained alpha": 14,
        "OFF sustained EW1no": 15,
        "OFF sustained - subtype unknown": 16,
        "OFF transient alpha": 17,
        "OFF transient medium RF": 18,
        "OFF transient small RF": 19,
        "OFF transient - subtype unknown": 20,
        "F-mini-ON": 21,
        "Local edge detector": 22,
        "UHD": 23,
        "HD1": 24,
        "HD2": 25,
        "F-mini-OFF": 26,
        "M1": 27,
        "M2": 28,
        "PixON": 29,
        "ON alpha": 30,
        "ON horizontal OS large RF": 31,
        "ON horizontal OS small RF": 32,
        "ON vertical OS large RF": 33,
        "ON vertical OS small RF": 34,
        "ON OS large RF - orientation unknown": 35,
        "ON OS small RF - orientation unknown": 36,
        "M6": 37,
        "ON transient EW6t": 38,
        "ON transient medium RF": 39,
        "ON transient small RF": 40,
        "ON transient - subtype unknown": 41,
        "Sustained suppressed-by-contrast strong surround EW28": 42,
        "Sustained suppressed-by-contrast no surround EW27": 43,
        "Bursty suppressed by contrast"
        : 44,
        "ON bursty": 45,
        "ON delayed": 46,
        "ON small OFF large": 47,
        "ON sustained - subtype unknown": 48,
        "Motion sensor": 49,
        "ON DS transient": 50,
        "unknown": 51,
    },
    # Abgeglichen mit Functional classes of rgctypes.org
    cell_class_to_id = {
        "OFF sustained": [10, 11, 12, 13, 14, 15],
        "OFF transient": [17, 18, 19, 20],
        "ON sustained": [27, 28, 29, 30, 48],
        "ON transient": [38, 39, 40]
    },
    cell_class_to_type = {
        "Transient": [17, 18, 19, 20, 38, 39, 40],
        "Sustained": [10, 11, 12, 13, 14, 15, 27, 28, 29, 30, 48]
    },
    cell_class_to_group = {
        "ON": [27, 28, 29, 30, 48, 38, 39, 40],
        "OFF": [10, 11, 12, 13, 14, 15, 17, 18, 19, 20]
    }
)


RetinalGanglionCellResearchCenterJuelich = CellMergeClass(
    cell_type_to_id = {
        "OFF Sustained": 0,
        "OFF Transient": 1,
        "ON-OFF": 2,
        "ON Sustained": 3,
        "ON Transient": 4
    },
    cell_class_to_id = {},
    cell_class_to_type = {
        "Transient": [1, 4],
        "Sustained": [0, 3]
    },
    cell_class_to_group = {
        "OFF": [0, 1],
        "ON": [3, 4]
    }
)


def rgc_logic_combination(path2valid_data: str, valid_file_name: str='results_cl.npy', show_plot: bool=False) -> None:
    """Post-Classification of Retinal Ganglion Cell-Type Classifier (RGC) after NN training
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
