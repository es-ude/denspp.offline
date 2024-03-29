from torch import nn
from package.dnn.pytorch.handler import Config_PyTorch, Config_Dataset

import package.dnn.models.spike_detection as models_sda
import package.dnn.models.autoencoder_cnn as models_ae
import package.dnn.models.autoencoder_class as models_class
import package.dnn.models.rgc_onoff_class as models_rgc_class
import package.dnn.models.decoding_utah as models_dec


config_data = Config_Dataset(
    # --- Settings of Datasets
    data_path='C:\HomeOffice\Data_Neurosignal\\00_Merged',
    #data_path='../2_Data/00_Merged_Datasets',
    #data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',
    #data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_file_name='2024-02-05_Dataset-KlaesNeuralDecoding.npy',
    # --- Data Augmentation
    data_do_addnoise_cluster=False,
    data_do_augmentation=False,
    data_num_augmentation=0,
    # --- Data Normalisation
    data_do_normalization=False,
    data_normalization_mode='CPU',  # 'CPU', 'FPGA'
    data_normalization_method='minmax',  # 'minmax', 'norm', 'zscore', 'medianmad', 'meanmad'
    data_normalization_setting='bipolar',  # 'bipolar', 'global', ''
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
    num_epochs=2,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_ae.cnn_ae_v4(32, 8),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=16,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_ae_class = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_class.classifier_ae_v1(8, 5),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=2,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_rgc = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_rgc_class.dnn_rgc_v2(32, 5),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=2,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_train_dec = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_dec.cnn_lstm_dec_v1(),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=20,
    batch_size=20,
    data_split_ratio=0.25,
    data_do_shuffle=True
)
