from torch import nn
from package.dnn.pytorch_control import Config_PyTorch
import package.dnn.models.autoencoder as ai_module


config_train_ae = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_ae_v2(),
    loss='MSE',
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    # --- Settings of Datasets
    data_path='../2_Data/00_Merged_Datasets',
    data_file_name='2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    # --- Dataset Reduction
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=50_000,
    data_exclude_cluster=[],
    data_sel_pos=[]
)
