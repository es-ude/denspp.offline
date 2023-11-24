import matplotlib.pyplot as plt
from torch import nn
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_confusion, plot_loss
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.rgc_onoff_class import prepare_training
import package.dnn.models.rgc_onoff_class as ai_module


num_output = 4

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_rgc_v2(input_size=32, output_size=num_output),
    loss_fn=nn.CrossEntropyLoss(),
    # loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=128,
    # --- Settings of Datasets
    data_path='data',
    # data_file_name='2023-11-16_rgc_onoff_fzj.mat',
    data_file_name='2023-11-17_Dataset-07_RGC_TDB_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=10000,
    # --- Dataset Preparation
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Main Program
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                               use_cell_bib=True, mode_classes=1)
    dataset_dict = dataset.frame_dict if dataset.cluster_name_available else []
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset, config_train
    epoch_acc = trainhandler.do_training()[0]

    # --- Post-Processing: Getting data, save and plot results
    data_result = trainhandler.do_validation_after_training(2)

    logsdir = trainhandler.get_saving_path()
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_confusion(data_result['valid_clus'], data_result['yclus'], path2save=logsdir, cl_dict=dataset_dict)
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'], path2save=logsdir, cl_dict=dataset_dict)

    plt.show(block=False)
    plt.close("all")
