import matplotlib.pyplot as plt
from torch import nn
from os.path import join
from scipy.io import loadmat
from package.plotting.plot_dnn import results_training, plot_statistic_data
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_autoencoder import *
from package.dnn.dataset.autoencoder import prepare_training
import package.dnn.models.autoencoder as ai_module

from scipy.io import loadmat


noise_std = 1
# 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
mode_train = 0
use_cell_bib = True
mode_cell_bib = 1
only_plot = False

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.cnn_ae_v3(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=150,
    batch_size=64,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',
    # data_path='../2_Data/00_Merged_Datasets',
    # data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=True,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Main program
if __name__ == "__main__":
    if not only_plot:
        print("\nTrain modules of end-to-end neural signal pre-processing frame-work (DeNSSP)")
        # --- Processing: Loading dataset and Do Training
        dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                                   use_cell_bib=use_cell_bib, mode_classes=mode_cell_bib,
                                   mode_train_ae=mode_train, do_classification=False,
                                   noise_std=noise_std)
        dataset_dict = dataset.frame_dict
        data_mean = dataset.frames_me
        trainhandler = pytorch_train(config_train)
        trainhandler.load_model()
        trainhandler.load_data(dataset)
        del dataset

        snr_train = trainhandler.do_training()
        logsdir = trainhandler.get_saving_path()
        data_result = trainhandler.do_validation_after_training()
    else:
        snr_train = list()
        logsdir = 'runs/20231201_165901_train_cnn_ae_v3/'
        data_result = loadmat(join(logsdir, 'results.mat'))
        data_mean = np.zeros(shape=(52, 32))
        data_mean = np.zeros(shape=(4,32))
        dataset_dict = dict()

    plt.close("all")
    results_training(
        path=logsdir, cl_dict=data_result['cl_dict'], feat=data_result['feat'],
        yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
        yclus=data_result['valid_clus'], snr=snr_train
    )
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=data_result['cl_dict'])

    plt.show(block=True)

