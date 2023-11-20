import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from torch import nn, from_numpy, load
from scipy.io import savemat

import package.plotting.plot_dnn as plt_spaike
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_autoencoder import *
from package.dnn.dataset.autoencoder import prepare_plotting, prepare_training
from package.dnn.data_preprocessing import calculate_frame_mean, calculate_frame_snr
import package.dnn.models.autoencoder as ai_module


# 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
mode_train = 0

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_ae_rgc_fzj_v2(),
    is_embedded=False,
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=10,
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

# --- Main programme
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train, mode_train_ae=mode_train)
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    loss, snr_train = trainhandler.do_training()

    # --- Post-Processing: Getting data from validation set for inference
    valid_dl = trainhandler.valid_loader[0]
    data_in, data_out, cluster_out, data_mean = prepare_plotting(valid_dl)
    del valid_dl

    # --- Post-Processing: Do the Inference with Best Model
    print(f"\nDoing the inference with validation data on best model")
    model_test = load(trainhandler.get_best_model()[0])
    model_in = from_numpy(data_in)
    feat_out, pred_out = model_test(model_in)
    feat0 = feat_out.detach().numpy()
    ypred0 = pred_out.detach().numpy()

    # --- Post-Processing: Calculating the improved SNR
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

    # --- Reducing data_mean and Saving data
    data_mean = calculate_frame_mean(data_in, cluster_out)

    logsdir = trainhandler.get_saving_path()
    savemat(join(logsdir, 'results.mat'),
            {"frames_in": data_in,
             "frames_out": data_out,
             "frames_mean": data_mean,
             "frames_pred": ypred0,
             "feat": feat0,
             "cluster": cluster_out,
             "config": config_train
             },
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
