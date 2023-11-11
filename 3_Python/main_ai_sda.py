from os.path import join
from glob import glob
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy, load
from scipy.io import savemat
import numpy as np

from package.dnn.pytorch_structure import pytorch_classifier
from package.dnn.dataset.spike_detection import prepare_plotting, prepare_training
import package.plotting.plot_dnn as plt_spaike
import package.dnn.models.spike_detection as ai_module


class Config_PyTorch:
    def __init__(self):
        # Settings of Models/Training
        self.model = ai_module.cnn_sda_v1
        # self.model = ai_module_embedded.dnn_dae_v2
        self.is_embedded = False
        self.loss_fn = nn.CrossEntropyLoss()
        self.num_kfold = 1
        self.num_epochs = 20
        self.batch_size = 256
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = 'SDA_Dataset.mat'
        self.data_split_ratio = 0.25
        self.data_do_shuffle = True
        self.data_do_augmentation = False
        self.data_num_augmentation = 2000
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        # Dataset Preparation
        self.data_exclude_cluster = []
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return optim.SGD(model.parameters(), lr=0.1)

    def get_topology(self, model) -> str:
        return model.out_modeltyp


# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

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
    trainhandler = pytorch_classifier(model_typ, model_name, model_settings)
    trainhandler.load_model(model, model_opt)
    trainhandler.load_data(dataset)

    loss, snr_train = trainhandler.do_training()

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
