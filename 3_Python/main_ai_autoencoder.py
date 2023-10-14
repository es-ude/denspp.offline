import os.path
import glob
import matplotlib.pyplot as plt
import torch
from scipy.io import savemat
import numpy as np

from package.metric import calculate_snr
from package.dnn.dataset_preparation import prepare_training
import package.dnn.plotting as plt_spaike
from package.dnn.pytorch_handler import training_pytorch
from package.dnn.ae_dataset import DatasetAE, prepare_plotting, get_dataloaders
import models.ae_topology as ai_module


# TODO: Add normal training of denoising autoencoder
# TODO: Data Normalization does not work very well
class Config_PyTorch:
    def __init__(self):
        # Settings of Models/Training
        self.model = ai_module.dnn_ae_v2
        # self.model = ai_module_embedded.dnn_dae_v2
        self.is_embedded = False
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs = 500
        self.batch_size = 256
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted'
        self.data_split_ratio = 0.25
        self.data_do_shuffle = True
        self.data_do_augmentation = True
        self.data_num_augmentation = 2000
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        # Dataset Preparation
        self.data_exclude_cluster = []
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return torch.optim.Adam(model.parameters())

    def get_topology(self, model) -> str:
        return model.out_modeltyp


def ae_addon(mode: int) -> str:
    addon = ' (Normal)'
    if mode == 1:
        addon = ' (Denoising, mean output)'
    elif mode == 2:
        addon = ' (Denoising, noise input)'

    return addon

# --- Hauptprogramm
if __name__ == "__main__":
    # 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
    mode_train = 1

    # --- Programme start
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    model_settings = Config_PyTorch()
    # --- Pre-Processing: Create NN
    model = model_settings.model()
    model_opt = model_settings.set_optimizer(model)
    model_name = model.out_modelname
    model_typ = model.out_modeltyp

    # --- Pre-Processing: Loading data and splitting into training and validation
    path = os.path.join(model_settings.data_path, model_settings.data_file_name)
    frames_in, frames_cluster, frames_mean = prepare_training(
        path=path,
        do_augmentation=model_settings.data_do_augmentation, num_new_frames=model_settings.data_num_augmentation,
        excludeCluster=model_settings.data_exclude_cluster, sel_pos=model_settings.data_sel_pos,
        do_norm=model_settings.data_do_normalization, do_zeroframes=model_settings.data_do_addnoise_cluster
    )
    dataset = DatasetAE(frames_in, frames_cluster, frames_mean, mode_train=mode_train)
    train_dl, valid_dl = get_dataloaders(
        dataset, batch_size=model_settings.batch_size,
        validation_split=model_settings.data_split_ratio,
        shuffle=model_settings.data_do_shuffle
    )

    # --- Processing: Do Training
    trainhandler = training_pytorch(model_typ, model_name, model_settings)
    trainhandler.model_addon = ae_addon(mode_train)
    trainhandler.load_model(model, model_opt)
    trainhandler.load_data(train_dl, valid_dl)

    snr_train = trainhandler.do_training()
    logsdir = trainhandler.path2save

    # --- Post-Processing: Getting data from validation set for plotting
    model_name_test = glob.glob(os.path.join(logsdir, 'model_*'))
    model_test = torch.load(model_name_test[0])

    data_in, data_out, cluster_out, data_mean = prepare_plotting(valid_dl)
    model_in = torch.from_numpy(data_in)
    feat_out, pred_out = model_test(model_in)
    feat0 = feat_out.detach().numpy()
    ypred0 = pred_out.detach().numpy()

    # --- Post-Processing: SNR improvement
    snr_in = []
    snr_out = []
    for i, frame in enumerate(data_in):
        snr_in.append(calculate_snr(frame, data_mean[i, :]))
        snr_out.append(calculate_snr(ypred0[i, :], data_mean[i, :]))

    snr_in = np.array(snr_in)
    snr_out = np.array(snr_out)
    snr_delta = snr_out - snr_in

    print(f"- SNR_in: {np.mean(snr_in):.2f} (mean) | {np.max(snr_in):.2f} (max) | {np.min(snr_in):.2f} (min)")
    print(f"- SNR_out: {np.mean(snr_out):.2f} (mean) | {np.max(snr_out):.2f} (max) | {np.min(snr_out):.2f} (min)")
    print(f"- SNR_inc: {np.mean(snr_delta): .2f} (mean)")

    # --- Saving data
    matdata = { "frames_in": data_in,
                "frames_out": data_out,
                "frames_mean": data_mean,
                "frames_pred": ypred0,
                "feat": feat0,
                "cluster": cluster_out,
                "config": model_settings
    }
    filename = 'results.mat'
    savemat(os.path.join(logsdir, filename), matdata)

    # --- Plotting
    plt_spaike.results_training(
        path=logsdir, feat=feat0,
        yin=data_in, ypred=ypred0, ymean=data_mean,
        cluster=cluster_out, snr=snr_train
    )
    plt.show(block=False)
    plt.close("all")

    print("\nLook data on TensorBoard -> open Terminal")
    print("Type in: tensorboard serve --logdir ./runs")
