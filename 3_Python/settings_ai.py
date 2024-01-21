from torch import nn
from package.dnn.pytorch_control import Config_PyTorch, Config_Dataset
import package.dnn.models.autoencoder as ae_models
import package.dnn.models.rgc_onoff_class as rgc_class_models
import package.dnn.models.spike_detection as sda_modes

config_dataset = Config_Dataset(
    # --- Settings of Datasets
    #data_path='../2_Data/00_Merged_Datasets',
    #data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_path="data",
    data_file_name="2023-11-24_Dataset-07_RGC_TDB_Merged",
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=100,
    data_do_normalization=True,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)


config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=ae_models.cnn_ae_v4(32, 6),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)
config_train_class = Config_PyTorch(
    # --- Settings of Models/Training
    model=rgc_class_models.dnn_rgc_v1(32, 4),
    loss='Cross Entropy',
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
    batch_size=512,
    data_split_ratio=0.25,
    data_do_shuffle=True
)
