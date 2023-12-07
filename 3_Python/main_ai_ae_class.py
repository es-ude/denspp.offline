import matplotlib.pyplot as plt
from torch import nn
from os.path import join
from scipy.io import loadmat
from package.plotting.plot_dnn import plot_statistic_data
from package.plotting.plot_metric import plot_loss, plot_confusion
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.autoencoder_class import prepare_training
import package.dnn.models.autoencoder as ai_module


noise_std = 1
# 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
mode_train = 1
use_cell_bib = False
mode_cell_bib = 0
only_plot = False

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.class_autoencoder_v1(),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    # --- Settings of Datasets
    # data_path='data',
    # data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=5000,
    data_do_normalization=True,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Main program
if __name__ == "__main__":
    if not only_plot:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # --- Processing: Loading dataset and Do Training
        dataset = prepare_training(path2data=config_train.get_path2data(), settings=config_train,
                                   path2model='runs/20231204_152721_train_cnn_ae_v3',
                                   use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib, noise_std=noise_std)
        data_mean = dataset.frames_me
        trainhandler = pytorch_train(config_train)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset

        epoch_acc = trainhandler.do_training()[0]
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
    else:
        epoch_acc = list()
        logsdir = 'runs/20231204_152721_train_cnn_ae_v3/'
        data_result = loadmat(join(logsdir, 'results.mat'))
        data_mean = np.zeros(shape=(52, 32))

    plt.close('all')
    # plot_loss(epoch_acc, 'Acc.', path2save=logsdir)
    plot_loss(epoch_acc, 'Acc.', path2save=logsdir, epoch_zoom=[80, ])
    plot_confusion(data_result['valid_clus'], data_result['yclus'],
                   path2save=logsdir, cl_dict=data_result['cl_dict'])
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=False)