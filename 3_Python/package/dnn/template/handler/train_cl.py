import matplotlib.pyplot as plt
from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_dataclass import (Config_PyTorch, DefaultSettingsDataset,
                                           Config_Dataset, DefaultSettingsTrainCE)
from package.plot.plot_dnn import plot_statistic_data
from package.plot.plot_metric import plot_confusion, plot_loss

from package.dnn.template.dataset.autoencoder import prepare_training
from package.dnn.pytorch.classifier import train_nn
import package.dnn.template.models.autoencoder_class as models


def do_train_classifier(settings: dnn_handler) -> None:
    """Training routine for Autoencoders
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
    default_train.model_name = models.classifier_ae_v1.__name__
    yaml_train = yaml_config_handler(default_train, yaml_name='Config_CL_Training')
    config_train = yaml_train.get_class(Config_PyTorch)
    del default_train, yaml_train

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data, do_classification=True)

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
    if settings.do_plot:
        plt.close('all')
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
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

    do_train_classifier(dnn_handler)
