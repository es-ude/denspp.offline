import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from torch import nn, from_numpy, load
from scipy.io import savemat

from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.rgc_onoff_class import prepare_plotting, prepare_training
import package.plotting.plot_dnn as plt_spaike
import package.dnn.models.rgc_onoff_class as ai_module


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_rgc_v1(),
    is_embedded=False,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=20,
    batch_size=256,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='2023-11-16_rgc_onoff_fzj.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Data Augmentation
    data_do_augmentation=True,
    data_num_augmentation=0,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=20000,
    # --- Dataset Preparation
    data_exclude_cluster=[1, 2, 3],
    data_sel_pos=[]
)

# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train)
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss, epoch_metric = trainhandler.do_training()

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
