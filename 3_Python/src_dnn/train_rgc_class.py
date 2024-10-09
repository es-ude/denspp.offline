from os import mkdir
from os.path import join, exists
from numpy import load

from package.yaml_handler import yaml_config_handler
from package.plot.plot_dnn import plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss
from package.data_call.call_cellbib import logic_combination
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch.classifier import train_nn
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE

from src_dnn.dataset.rgc_classification import prepare_training
import src_dnn.models.rgc_onoff_class as models


def rgc_logic_combination(path2valid_data: str, valid_file_name='results_class.npy', show_plot=False) -> None:
    """Post-Classification of Retinal Ganglion Celltype Classifier (RGC) after NN training
    Args:
        path2valid_data:    Path to validation data (generated after training)
        valid_file_name:    Filename of validation data [Default: 'results_class.npy']
        show_plot:          Showing all plots
    Return:
        None
    """
    data_result = load(join(path2valid_data, valid_file_name), allow_pickle=True).item()
    path2save = join(path2valid_data, 'logic_comb')
    if not exists(path2save):
        mkdir(path2save)

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


def do_train_rgc_class(dnn_trainhandler: dnn_handler) -> None:
    """Training routine for classifying RGC ON/OFF and Transient/Sustained Types (Classification)
    Args:
        dnn_trainhandler: Handler for configuring the routine selection to train deep neural networks
    """
    # --- Loading the YAML files
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    default_data.data_file_name = '2023-11-24_Dataset-07_RGC_TDB_Merged.mat'
    yaml_data = yaml_config_handler(default_data, 'config', 'Config_RGC_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del default_data, yaml_data

    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.dnn_rgc_v1.__name__
    yaml_train = yaml_config_handler(default_train, 'config', 'Config_RGC_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data)
    frame_dict = dataset.frame_dict

    trainhandler = train_nn(config_train, config_data)
    model = models.models_available.build_model(config_train.model_name)
    trainhandler.load_model(model)
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if dnn_trainhandler.do_plot:
        # --- Plotting full model
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=logsdir, cl_dict=frame_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=frame_dict)

        # --- Plotting reduced model (ON/OFF and Transient/Sustained)
        rgc_logic_combination(logsdir, show_plot=dnn_trainhandler.do_block)
