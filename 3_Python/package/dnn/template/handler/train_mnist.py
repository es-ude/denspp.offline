import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsTrainCE, DefaultSettingsTrainMSE,
                                           Config_Dataset, DefaultSettingsDataset)
from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
from package.plot.plot_metric import plot_confusion, plot_loss

from package.dnn.template.dataset.mnist import prepare_training
from package.dnn.pytorch.autoencoder import train_nn as train_nn_ae
from package.dnn.pytorch.classifier import train_nn as train_nn_cl
import package.dnn.template.models.mnist as models


def do_train_cl(do_plot=True, do_block=True) -> None:
    """Training routine for classifying neural activations
    Args:
        do_plot:            Plotting the results
        do_block:           Blocking the plots for user interactions, otherwise only saving
    Returns:
        None
    """
    # --- Loading the YAML files
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    yaml_data = yaml_config_handler(default_data, yaml_name='Config_MNIST_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del default_data, yaml_data

    default_train = DefaultSettingsTrainCE
    default_train.model_name = models.mnist_mlp_cl_v1.__name__
    yaml_train = yaml_config_handler(default_train, yaml_name='Config_MNIST_TrainCL')
    config_train = yaml_train.get_class(Config_PyTorch)
    model = models.models_available.build_model(config_train.model_name)
    del default_train, yaml_train

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data, True)

    trainhandler = train_nn_cl(config_train, config_data)
    trainhandler.load_model(model)
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close('all')
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, show_plots=do_block)
    print("\nThe End")


def do_train_ae(do_plot=True, do_block=True) -> None:
    """Training routine for training an autoencoder
    Args:
        do_plot:            Plotting the results
        do_block:           Blocking the plots for user interactions, otherwise only saving
    Returns:
        None
    """
    # --- Loading the YAML files
    default_data = DefaultSettingsDataset
    default_data.data_path = 'data'
    yaml_data = yaml_config_handler(default_data, aml_name='Config_MNIST_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)
    del default_data, yaml_data

    default_train = DefaultSettingsTrainMSE
    default_train.model_name = models.mnist_mlp_ae_v1.__name__
    yaml_train = yaml_config_handler(default_train, yaml_name='Config_MNIST_TrainAE')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data,False)

    trainhandler = train_nn_ae(config_train, config_data)
    model = models.models_available.build_model(config_train.model_name)
    trainhandler.load_model(model)
    trainhandler.load_data(dataset)
    del dataset
    epoch_loss = trainhandler.do_training()[-1][0]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
        plt.close('all')
        plot_loss(epoch_loss, 'Loss', path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], "_input", path2save=logsdir)
        plot_mnist_graphs(data_result['pred'], data_result['valid_clus'], "_predicted", path2save=logsdir,
                          show_plot=do_block)
    print("\nThe End")


if __name__ == "__main__":
    do_train_cl(True, False)
    do_train_ae(True, True)
