from torch import nn
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.models.mnist as models


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='data',
    data_file_name="",
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=True,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=0,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train_cl = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.mnist_mlp_cl_v1(),
    loss='Cross Entropy Loss',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    patience=20,
    num_epochs=5,
    batch_size=512,
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
    patience=20,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_cl(do_plot=True, do_block=True) -> None:
    """Training routine for classifying neural activations
    Args:
        do_plot: Plotting the results
        do_block: Blocking the plots for user interactions, otherwise only saving
    """
    from package.dnn.dataset.mnist import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
    from package.plot.plot_metric import plot_confusion, plot_loss

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
        # --- Plotting full model
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, show_plots=do_block)


def do_train_ae(do_plot=True, do_block=True) -> None:
    """Training routine for training an autoencoder
    Args:
        do_plot: Plotting the results
        do_block: Blocking the plots for user interactions, otherwise only saving
    """
    from package.dnn.dataset.mnist import prepare_training
    from package.dnn.pytorch.autoencoder import train_nn
    from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
    from package.plot.plot_metric import plot_loss

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data.data_path, config_data.data_do_normalization, False)
    trainhandler = train_nn(config_train_ae, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_loss = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if do_plot:
        # --- Plotting full model
        plot_loss(epoch_loss, 'Loss', path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plot_mnist_graphs(data_result['input'], data_result['valid_clus'], "_input", path2save=logsdir)
        plot_mnist_graphs(data_result['pred'], data_result['valid_clus'], "_predicted",
                          path2save=logsdir, show_plot=do_block)
