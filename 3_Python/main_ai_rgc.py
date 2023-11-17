import numpy as np
from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy, load
from scipy.io import savemat

from package.dnn.pytorch_classification import *
from package.dnn.dataset.rgc_onoff_class import prepare_plotting, prepare_training
import package.plotting.plot_dnn as plt_spaike
import package.dnn.models.rgc_onoff_class as ai_module


class Config_PyTorch:
    def __init__(self):
        # Settings of Models/Training
        self.model = ai_module.dnn_rgc_v1
        # self.model = ai_module_embedded.dnn_dae_v2
        self.is_embedded = False
        # self.loss_fn = nn.NLLLoss() # Negative Log-Likelihood Loss function
        # self.loss_fn = nn.L1Loss() # MAE
        # self.loss_fn = nn.MSELoss()
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.BCELoss(reduction='mean')
        self.num_kfold = 1
        self.num_epochs = 500
        self.batch_size = 256
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = '2023-11-16_rgc_onoff_fzj.mat'
        self.data_split_ratio = 0.25
        self.data_do_shuffle = True
        self.data_do_augmentation = False
        self.data_num_augmentation = 2000
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        self.data_do_reduce_samples_per_cluster = True
        self.data_num_samples_per_cluster = 20000
        # Dataset Preparation
        self.data_exclude_cluster = [1, 2, 3]
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return optim.SGD(model.parameters(), lr=0.1)

    def get_topology(self, model) -> str:
        return model.out_modeltyp


# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    output_ch = 4

    # --- Pre-Processing: Create NN
    model_settings = Config_PyTorch()
    model = model_settings.model()
    model_opt = model_settings.set_optimizer(model)
    model_name = model.out_modelname
    model_typ = model.out_modeltyp

    # --- Pre-Processing: Loading dataset
    path = join(model_settings.data_path, model_settings.data_file_name)
    dataset = prepare_training(path=path, settings=model_settings)

    # --- Processing: Do Training
    trainhandler = pytorch_train(model_typ, model_name, model_settings)
    trainhandler.load_model(model, model_opt)
    trainhandler.load_data(dataset)

    loss, epoch_metrics = trainhandler.do_training()

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
