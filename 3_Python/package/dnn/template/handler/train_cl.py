import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsDataset,
                                           Config_Dataset, DefaultSettingsTrainCE)
from package.plot.plot_dnn import plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss

from package.dnn.template.dataset.autoencoder import prepare_training
from package.dnn.pytorch.classifier import train_nn
import package.dnn.template.models.spike_classifier as models


def do_train_cl(settings: dnn_handler) -> None:
    """Training routine for Classification DL models
    Args:
        settings:           Handler for configuring the routine selection for train deep neural networks
    Returns:
        None
    """
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(DefaultSettingsDataset, yaml_name='Config_CL_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del yaml_data

    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.spike_cl_v1.__name__
    yaml_train = yaml_config_handler(default_train, yaml_name='Config_CL_Training')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data, do_classification=True)

    train_handler = train_nn(config_train, config_data)
    model = models.models_available.build_model(config_train.model_name)
    train_handler.load_model(model)
    train_handler.load_data(dataset)
    del dataset
    metrics = train_handler.do_training()

    # --- Post-Processing: Getting data, save and plot results
    logsdir = train_handler.get_saving_path()
    data_result = train_handler.do_validation_after_training()
    del train_handler

    # --- Plotting
    if settings.do_plot:
        plt.close('all')
        used_first_fold = [key for key in metrics.keys()][0]

        plot_loss(metrics[used_first_fold]['train_acc'], metrics[used_first_fold]['valid_acc'],
                  type='Acc.', path2save=logsdir)
        plot_loss(metrics[used_first_fold]['train_loss'], metrics[used_first_fold]['valid_loss'],
                  type=f'{config_train.loss} (CL)', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=logsdir, cl_dict=data_result['cl_dict'])
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'], show_plot=settings.do_block)
    print("The End")


if __name__ == "__main__":
    dnn_handler = dnn_handler(
        mode_train_dnn=0,
        do_plot=True,
        do_block=True
    )
    do_train_cl(dnn_handler)
