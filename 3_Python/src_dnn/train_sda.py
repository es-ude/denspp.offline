from torch import nn
import matplotlib.pyplot as plt

from package.yaml_handler import yaml_config_handler
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import src_dnn.models.spike_detection as models


config_train_sda = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.dnn_sda_v1(16, 5),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    patience=20,
    num_epochs=2,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def dnn_train_sda(dnn_handler: dnn_handler, sda_threshold: int) -> None:
    """Training routine for Spike Detection
    Args:
        dnn_handler: Handler for configurating the routine selection for train deep neural networks
        sda_threshold: Threshold value for identifying a spike event
    """
    from src_dnn.dataset.spike_detection import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # --- Loading the YAML files
    yaml_data = yaml_config_handler(models.Recommended_Config_DatasetSettings, yaml_name='Config_SDA_Dataset')
    config_data = yaml_data.get_class(Config_Dataset)

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(config_data, sda_threshold)
    data_dict = dataset.sda_dict
    trainhandler = train_nn(config_train_sda, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training()
    logsdir = trainhandler.get_saving_path()
    del trainhandler

    # --- Plotting
    if dnn_handler.do_plot:
        plt.close('all')
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir,
                            cl_dict=data_dict, show_plot=dnn_handler.do_block)
