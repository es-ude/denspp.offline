from torch import nn
from package.dnn.dnn_handler import dnn_handler
from package.dnn.pytorch_handler import Config_PyTorch, Config_Dataset
import package.dnn.template.models.autoencoder_class as models


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_addnoise_cluster=False,
    # --- Data Normalization
    data_do_normalization=False,
    data_normalization_mode='CPU',
    data_normalization_method='minmax',
    data_normalization_setting='bipolar',
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=models.classifier_ae_v1(32, 5),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    patience=10,
    data_split_ratio=0.25,
    data_do_shuffle=True
)


def do_train_classifier(dnn_handler: dnn_handler) -> None:
    """Training routine for Autoencoders
    Args:
        dnn_handler: Handler for configurating the routine selection for train deep neural networks
        mode_cell_bib: If the dataset contains a cell library then the mode can be choicen (0: Deactivated, 1: All, 2-...: Reduced) [default: 0]
        do_plot: Doing the plots during the training routine
        block_plot: Blocking the plot outputs if do_plot is active
    """
    from package.dnn.template.dataset.autoencoder import prepare_training
    from package.dnn.pytorch.classifier import train_nn
    from package.plot.plot_dnn import plot_statistic_data
    from package.plot.plot_metric import plot_confusion, plot_loss

    use_cell_bib = not (dnn_handler.mode_cell_bib == 0)
    use_cell_mode = 0 if not use_cell_bib else dnn_handler.mode_cell_bib - 1

    # ---Loading Data, Do Training and getting the results
    dataset = prepare_training(config_data, use_cell_bib=use_cell_bib, mode_classes=use_cell_mode,
                               do_classification=True)
    trainhandler = train_nn(config_train, config_data)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    epoch_acc = trainhandler.do_training()[-1]

    # --- Post-Processing: Getting data, save and plot results
    logsdir = trainhandler.get_saving_path()
    data_result = trainhandler.do_validation_after_training()
    del trainhandler

    # --- Plotting
    if dnn_handler.do_plot:
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
        plot_confusion(data_result['valid_clus'], data_result['yclus'],
                       path2save=logsdir, cl_dict=data_result['cl_dict'])
        plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                            path2save=logsdir, cl_dict=data_result['cl_dict'], show_plot=dnn_handler.do_block)
    print("The End")


if __name__ == "__main__":
    dnn_handler = dnn_handler(
        mode_dnn=1,
        mode_cellbib=0,
        do_plot=True,
        do_block=True
    )

    do_train_classifier(dnn_handler)
