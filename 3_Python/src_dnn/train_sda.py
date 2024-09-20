from torch import nn
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import src_dnn.models.spike_detection as models_sda


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    #data_file_name='2023-06-30_Dataset03_SimDaten_Quiroga2020_Sorted',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='',
    data_normalization_method='',
    data_normalization_setting='',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train_sda = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_sda.dnn_sda_v1(16, 5),
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
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=data_dict)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir,
                            cl_dict=data_dict, show_plot=dnn_handler.do_block)
