from torch import nn
import matplotlib.pyplot as plt
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import src_dnn.models.mnist_gki_class as models


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../data',
    data_file_name='',
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
    data_num_samples_per_cluster=5_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.mnist_gki_v8(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_cl = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.mnist_gki_v7(),
    loss='Cross Entropy Loss',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_ae(settings: dnn_handler) -> None:
    """Training routine for Autoencoders
    Args:
        settings: Handler for configuring the routine selection for train deep neural networks
    """
    from package.dnn.dataset.mnist import prepare_training
    from package.dnn.pytorch.autoencoder import train_nn
    from package.plot.plot_dnn import plot_statistic_data, plot_mnist_graphs
    from package.plot.plot_metric import plot_loss

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(config_data.data_path, config_data.data_do_normalization, False)
    trainhandler = train_nn(config_train_ae, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    loss_ae, snr_train = trainhandler.do_training()[-1]

    # --- Post-Processing: Validation after training
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()

    # --- Plotting and Ending
    if settings.do_plot:
        plt.close("all")
        plot_mnist_graphs(data_result["input"], label=data_result['valid_clus'], title='Input', path2save=logsdir)
        plot_mnist_graphs(data_result["pred"], label=data_result['valid_clus'], title='Reconstructed', path2save=logsdir)
        plot_loss(loss_ae, 'Acc.', path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'])
        plt.show(block=settings.do_block)


def do_train_cl(settings: dnn_handler) -> None:
    """Training routine for classifying neural activations
    Args:
        settings: Handler for configuring the routine selection for train deep neural networks
    """
    from package.dnn.dataset.mnist import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data.data_path, config_data.data_do_normalization, True)
    frame_dict = dataset.frame_dict
    num_output = len(frame_dict)
    trainhandler = train_nn(config_train_cl, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training(num_output)
    del trainhandler

    # --- Plotting
    if settings.do_plot:
        plt.close('all')
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=logsdir, cl_dict=frame_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=frame_dict)
        plt.show(block=settings.do_block)


if __name__ == "__main__":
    dnn_settings = dnn_handler(
        mode_dnn=0,
        mode_cellbib=0,
        do_plot=True,
        do_block=False
    )

    # do_train_cl(dnn_settings)
    do_train_ae(dnn_settings)
    print("Done")
