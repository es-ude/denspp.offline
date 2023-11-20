import numpy as np
from glob import glob
from os.path import join
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy, load
from scipy.io import savemat

from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_classification import *
from package.dnn.dataset.spike_detection import prepare_plotting, prepare_training
import package.plotting.plot_dnn as plt_spaike
import package.dnn.models.spike_detection as ai_module


config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.cnn_sda_v1(),
    loss_fn=nn.CrossEntropyLoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
    batch_size=256,
    # --- Settings of Datasets
    data_path='data',
    data_file_name='SDA_Dataset.mat',
    data_split_ratio=0.25,
    data_do_shuffle=True,
    # --- Settings for Data Augmentation
    data_do_augmentation=False,
    data_num_augmentation=2000,
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=True,
    data_num_samples_per_cluster=20000,
    # --- Dataset Preparation
    data_exclude_cluster=[],
    data_sel_pos=[]
)

# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading Data and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train, threshold=2)
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss, epoch_metric = trainhandler.do_training()

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
