import matplotlib.pyplot as plt
from torch import nn
from scipy.io import loadmat
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_confusion, plot_loss
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.rgc_onoff_class import prepare_training
import package.dnn.models.rgc_onoff_class as ai_module

only_plot = False
num_output = 4
mode_celllib_dict = 1

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_rgc_v2(input_size=32, output_size=num_output),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    #loss='Neg. Like-Lihood',
    #loss_fn=nn.NLLLoss(),
    #loss='KLDiv',
    #loss_fn=nn.KLDivLoss(reduction='batchmean'),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=300,
    batch_size=1024,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',
    # data_path='../2_Data/00_Merged_Datasets',
    # data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.2,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=True,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=416_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Main Program
if __name__ == "__main__":
    if not only_plot:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # ---Loading Data, Do Training and getting the results
        dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                                   use_cell_bib=True, mode_classes=mode_celllib_dict)
        trainhandler = pytorch_train(config_train)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset, config_train

        epoch_acc = trainhandler.do_training()[0]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training(2)
    else:
        epoch_acc = list()
        logsdir = 'runs/20231204_004158_train_rgc_class_v2/'
        data_result = loadmat(join(logsdir, 'results.mat'))

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[500, ])
    plot_confusion(data_result['valid_clus'], data_result['yclus'],
                   path2save=logsdir, cl_dict=data_result['cl_dict'])
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)
