from torch import nn
import matplotlib.pyplot as plt

from package.yaml_handler import yaml_config_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.template.models.mnist as models


config_train_cl = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.mnist_mlp_cl_v1(),
    loss='Cross Entropy Loss',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=5,
    batch_size=256,
    patience=10,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.mnist_mlp_ae_v1(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    patience=20,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_cl(do_plot=True, do_block=True) -> None:
    """Training routine for classifying neural activations
    Args:
        do_plot:            Plotting the results
        do_block:           Blocking the plots for user interactions, otherwise only saving
    Returns:
        None
    """
    from package.dnn.template.dataset.mnist import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
    from package.plot.plot_metric import plot_confusion, plot_loss

    print("\nTraining routine for MNIST classification")
    # --- Loading the YAML files
    yaml_data = yaml_config_handler(models.Recommended_Config_DatasetSettings, yaml_name='Config_MNIST_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data.data_path, config_data.data_do_normalization, True)
    trainhandler = train_nn(config_train_cl, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
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
    from package.dnn.template.dataset.mnist import prepare_training
    from package.dnn.pytorch.autoencoder import train_nn
    from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
    from package.plot.plot_metric import plot_loss

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data.data_path, config_data.data_do_normalization, False)
    trainhandler = train_nn(config_train_ae, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_loss = trainhandler.do_training()[-1][0]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
        plot_loss(epoch_loss, 'Loss', path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], "_input", path2save=logsdir)
        plot_mnist_graphs(data_result['pred'], data_result['valid_clus'], "_predicted", path2save=logsdir,
                          show_plot=do_block)
    print("\nThe End")
