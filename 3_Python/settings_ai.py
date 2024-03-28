from torch import nn
from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset
import package.dnn.models.autoencoder_cnn as models_ae
import package.dnn.models.autoencoder_class as models_class
import package.dnn.models.decoding as models_dec
import package.dnn.models.autoencoder_embedded as models_ae_embedded

config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_ae.cnn_ae_v4(32, 8),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=1,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)

config_dataset = Config_Dataset(
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-11-24_Dataset-07_RGC_TDB_Merged.mat',  # settings here!
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=100,
    data_do_normalization=True,
    data_normalization_mode='FPGA',  # 'CPU', ''
    data_normalization_method='',  # 'minmax', 'norm', 'zscore', 'medianmad', 'meanmad'
    data_normalization_setting='combined',  # 'bipolar', 'global', ''
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)

config_train_class = Config_PyTorch(
    # --- Settings of Models/Training
    model=models_class.classifier_ae_v1(12, 3),
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
