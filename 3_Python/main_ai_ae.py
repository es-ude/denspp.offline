import matplotlib.pyplot as plt
from torch import nn
from package.plotting.plot_dnn import results_training, plot_statistic_data
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_autoencoder import *
from package.dnn.dataset.autoencoder import prepare_training
import package.dnn.models.autoencoder as ai_module


noise_std = 1
# 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
mode_train = 2

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_ae_v2(),
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=5,
    batch_size=256,
    # --- Settings of Datasets
    # data_path='data',
    # data_file_name='2023-11-16_rgc_onoff_fzj.mat',
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=20000,
    # --- Dataset Preparation
    data_exclude_cluster=[4],
    data_sel_pos=[]
)

# --- Main program
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                               use_cell_bib=False, mode_classes=2,
                               mode_train_ae=mode_train, do_classification=False,
                               noise_std=noise_std)
    dataset_dict = dataset.frame_dict
    data_mean = dataset.frames_me
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    snr_train = trainhandler.do_training()

    # --- Post-Processing: Getting data, save and plot results

if False:
    data_result = trainhandler.do_validation_after_training()

    logsdir = trainhandler.get_saving_path()
    results_training(
        path=logsdir, feat=data_result['feat'],
        yin=data_result['input'], ypred=data_result['pred'], ymean=data_mean,
        yclus=data_result['valid_clus'], snr=snr_train
    )
    plot_statistic_data(data_result['train_clus'], data_result['valid_clus'],
                        path2save=logsdir, cl_dict=dataset_dict)

    plt.show(block=False)
    plt.close("all")
