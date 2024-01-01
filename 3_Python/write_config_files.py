from torch import nn
from package.dnn.pytorch_autoencoder import *

import package.dnn.models.autoencoder as ai_module

config_loss_dict = {'Autoencoder': 'MSE', 'Classification': 'Cross Entropy'}
config_loss_fn_dict = {'Autoencoder': nn.MSELoss(), 'Classification': nn.CrossEntropyLoss()}


class ConfigFiles (modeltyp='Autoencoder'):
    def __init(self, modeltyp):
        self.modeltyp = modeltyp

    def generate_config(self):
        config = Config_PyTorch(
            model=ai_module.dnn_ae_v2(),
            loss=config_loss_dict[self.modeltyp],
            loss_fn=config_loss_fn_dict[self.modeltyp],
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
            data_do_normalization=True,
            data_do_addnoise_cluster=False,
            data_do_reduce_samples_per_cluster=False,
            data_num_samples_per_cluster=20000,
            # --- Dataset Preparation
            data_exclude_cluster=[4],
            data_sel_pos=[]
        )

        return config
