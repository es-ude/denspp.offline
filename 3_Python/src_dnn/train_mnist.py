from torch import nn
import matplotlib.pyplot as plt
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.models.mnist as ae_models


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-06-30_Dataset03_SimDaten_Quiroga2020_Sorted.mat',
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=749,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=5_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train_cl = Config_PyTorch(
    # --- Settings of Models/Training
    model=ae_models.mlp_class_v1(),
    loss='Cross Entropy Loss',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=5,
    batch_size=256,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_cl(settings: dnn_handler) -> None:
    """Training routine for classifying neural activations
    Args:
        settings: Handler for configurating the routine selection for train deep neural networks
    """
    from package.dnn.dataset.mnist import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSPP)")

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data.data_path, True)
    num_output = 10
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
        # --- Plotting full model
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir)
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir)
        plt.show(block=settings.do_block)
    print("\nThe End")
