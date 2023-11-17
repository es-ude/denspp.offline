import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy, load
from scipy.io import savemat

import package.plotting.plot_dnn as plt_spaike
from package.dnn.pytorch_autoencoder import *
from package.dnn.dataset.autoencoder import prepare_plotting, prepare_training
from package.dnn.data_preprocessing import calculate_frame_mean, calculate_frame_snr
import package.dnn.models.autoencoder as ai_module


class Config_PyTorch:
    def __init__(self):
        # Settings of Models/Training
        self.model = ai_module.dnn_ae_rgc_fzj_v2
        # self.model = ai_module_embedded.dnn_dae_v2
        self.is_embedded = False
        # self.loss_fn = torch.nn.L1Loss()
        self.loss_fn = nn.MSELoss()
        self.num_kfold = 1
        self.num_epochs = 100
        self.batch_size = 256
        # Settings of Datasets
        self.data_path = 'data'
        self.data_file_name = '2023-11-16_rgc_onoff_fzj.mat'
        # self.data_path = '../2_Data/00_Merged_Datasets'
        # self.data_file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted'
        self.data_split_ratio = 0.25
        self.data_do_shuffle = True
        self.data_do_augmentation = True
        self.data_num_augmentation = 0
        self.data_do_normalization = False
        self.data_do_addnoise_cluster = False
        self.data_do_reduce_samples_per_cluster = True
        self.data_num_samples_per_cluster = 20000
        # Dataset Preparation
        self.data_exclude_cluster = [1, 2, 3]
        self.data_sel_pos = []

    def set_optimizer(self, model):
        return optim.Adam(model.parameters())

    def get_topology(self, model) -> str:
        return model.out_modeltyp


def ae_addon(mode: int) -> str:
    addon = ' (Normal)'
    if mode == 1:
        addon = ' (Denoising, mean output)'
    elif mode == 2:
        addon = ' (Denoising, noise input)'

    return addon


# --- Main programme
if __name__ == "__main__":
    # 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
    mode_train = 0

    # --- Programme start
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
    dataset = prepare_training(path=path, settings=model_settings, mode_train_ae=mode_train)

    # --- Processing: Do Training
    trainhandler = pytorch_train(model_typ, model_name, model_settings)
    trainhandler.model_addon = ae_addon(mode_train)
    trainhandler.load_model(model, model_opt)
    trainhandler.load_data(dataset)

    loss, snr_train = trainhandler.do_training()
    logsdir = trainhandler.get_saving_path()

    # --- Post-Processing: Getting data from validation set for plotting
    valid_dl = trainhandler.valid_loader[0]
    data_in, data_out, cluster_out, data_mean = prepare_plotting(valid_dl)
    del valid_dl

    model_name_test = glob(join(logsdir, 'model_fold*.pth'))
    model_test = load(model_name_test[0])

    # Doing the inference of best model
    print(f"\nDoing the inference with validation data on best model")
    model_in = from_numpy(data_in)
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

    # --- Reducing data_mean
    data_mean = calculate_frame_mean(data_in, cluster_out)

    # --- Saving data
    matdata = {"frames_in": data_in,
               "frames_out": data_out,
               "frames_mean": data_mean,
               "frames_pred": ypred0,
               "feat": feat0,
               "cluster": cluster_out,
               "config": model_settings
               }
    filename = 'results.mat'
    savemat(join(logsdir, filename), matdata,
            do_compression=True,
            long_field_names=True)

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
