from os import mkdir
from os.path import join, exists
from numpy import load

from package.yaml_handler import yaml_config_handler
from package.plot.plot_metric import plot_confusion
from package.data_call.call_cellbib import logic_combination
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import src_dnn.models.rgc_onoff_class as models_rgc


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_rgc.cnn_rgc_onoff_v1(32, 4),
    loss='Cross Entropy',
    optimizer='Adam',
    num_kfold=1,
    patience=20,
    num_epochs=100,
    batch_size=256,
    data_split_ratio=0.2,
    data_do_shuffle=True
)


def rgc_logic_combination(logsdir: str, valid_file_name='results_class.npy', show_plot=False) -> None:
    """"""
    data_result = load(join(logsdir, valid_file_name), allow_pickle=True).item()
    path2save = join(logsdir, 'logic_comb')
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
    from src_dnn.dataset.rgc_classification import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # --- Loading the YAML files
    yaml_handler = yaml_config_handler(models_rgc.Recommended_Config_DatasetSettings, 'config', 'Config_RGC_Dataset')
    config_data = yaml_handler.get_class(Config_Dataset)

    use_cell_bib = not (dnn_trainhandler.mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else dnn_trainhandler.mode_cell_bib - 1

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data, use_cell_bib=use_cell_bib, mode_classes=use_cell_mode)
    frame_dict = dataset.frame_dict
    trainhandler = train_nn(config_train, config_data)
    trainhandler.load_model()
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
