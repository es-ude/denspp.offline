from os import mkdir
from os.path import join, exists
from numpy import load

from package.data_call.call_cellbib import logic_combination
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import Config_ML_Pipeline
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE
from package.dnn.pytorch_pipeline import do_train_classifier

from package.plot.plot_metric import plot_confusion
from src_dnn.dataset.rgc_classification import prepare_training
import src_dnn.models.rgc_onoff_class as models


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


def do_train_rgc_class(settings: Config_ML_Pipeline, yaml_name_index='Config_RGC') -> None:
    """Training routine for classifying RGC ON/OFF and Transient/Sustained Types (Classification)
    Args:
        settings:           Handler for configuring the routine selection to train deep neural networks
        yaml_name_index:    Index of yaml file name
    """
    # --- Loading the YAML file: Dataset
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    default_data.data_file_name = '2023-11-24_Dataset-07_RGC_TDB_Merged.mat'
    yaml_data = yaml_config_handler(default_data, settings.get_path2config, f'{yaml_name_index}_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Loading the YAML file: Model training
    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.dnn_rgc_v1.__name__
    yaml_train = yaml_config_handler(default_train, settings.get_path2config, f'{yaml_name_index}_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)

    # ---Loading Data, Build Model and Do Training
    used_dataset = prepare_training(config_data)
    used_model = models.models_available.build_model(config_train.model_name)
    _, _, path2folder = do_train_classifier(
        config_ml=settings, config_data=config_data, config_train=config_train,
        used_dataset=used_dataset, used_model=used_model, path2save=''
    )

    # --- Plotting reduced model (ON/OFF and Transient/Sustained)
    if settings.do_plot:
        rgc_logic_combination(path2folder, show_plot=settings.do_block)


if __name__ == "__main__":
    from package.dnn.dnn_handler import DefaultSettings_MLPipe
    do_train_rgc_class(DefaultSettings_MLPipe)
