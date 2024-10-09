import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import Config_Dataset, DefaultSettingsDataset, Config_PyTorch, DefaultSettingsTrainCE
from package.dnn.pytorch.classifier import train_nn
from package.plot.plot_dnn import plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss

from src_dnn.dataset.spike_detection import prepare_training
import src_dnn.models.spike_detection as models


def dnn_train_sda(settings: dnn_handler, sda_threshold=4) -> None:
    """Training routine for Spike Detection
    Args:
        settings:       Handler for configurating the routine selection for train deep neural networks
        sda_threshold:  Threshold value for identifying a spike event
    Return:
        None
    """
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_SDA_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.dnn_sda_v1.__name__
    yaml_train = yaml_config_handler(default_train, 'config', 'Config_SDA_Train')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(config_data, sda_threshold)
    data_dict = dataset.sda_dict

    trainhandler = train_nn(config_train, config_data)
    model_used = models.models_available.build_model(config_train.model_name)
    trainhandler.load_model(model_used)
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training()
    logsdir = trainhandler.get_saving_path()
    del trainhandler

    # --- Plotting
    if settings.do_plot:
        plt.close('all')
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir,
                            cl_dict=data_dict, show_plot=settings.do_block)
