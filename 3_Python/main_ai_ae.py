from os.path import join
import matplotlib.pyplot as plt
from torch import nn, from_numpy, load
from scipy.io import savemat

from package.plotting.plot_dnn import results_training, plot_statistic_data
from package.dnn.pytorch_control import Config_PyTorch
from package.dnn.pytorch_autoencoder import *
from package.dnn.dataset.autoencoder import prepare_plotting, prepare_training
import package.dnn.models.autoencoder as ai_module


noise_std = 1
# 0 = normal autoencoder, 1 = denoising AE (mean), 2 = denoising AE (more noise input)
mode_train = 2

config_train = Config_PyTorch(
    # --- Settings of Models/Training
    model=ai_module.dnn_ae_v2(),
    loss_fn=nn.MSELoss(),
    optimizer='Adam',
    num_kfold=1,
    num_epochs=100,
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
    data_do_normalization=False,
    data_do_addnoise_cluster=False,
    data_do_reduce_samples_per_cluster=False,
    data_num_samples_per_cluster=20000,
    # --- Dataset Preparation
    data_exclude_cluster=[4],
    data_sel_pos=[]
)

# --- Main programme
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # --- Processing: Loading dataset and Do Training
    dataset = prepare_training(path=config_train.get_path2data(), settings=config_train,
                               mode_train_ae=mode_train, do_classification=False,
                               noise_std=noise_std)
    dataset_dict = dataset.frame_dict
    data_mean = dataset.frames_me
    trainhandler = pytorch_train(config_train)
    trainhandler.load_model()
    trainhandler.load_data(dataset)
    del dataset
    snr_train = trainhandler.do_training()

    # --- Post-Processing: Getting data from validation set for inference
    take_fold = 0
    data_in, data_out, cluster_out, data_mean0 = prepare_plotting(trainhandler.valid_loader[take_fold])
    yclus = prepare_plotting(trainhandler.train_loader[take_fold])[2]

    # --- Post-Processing: Do the Inference with Best Model
    print(f"\nDoing the inference with validation data on best model")
    model_test = load(trainhandler.get_best_model()[take_fold])
    feat_out, pred_out = model_test(from_numpy(data_in))
    feat_out = feat_out.detach().numpy()
    pred_out = pred_out.detach().numpy()

    # --- Post-Processing: Calculating the improved SNR
    snr_in = []
    snr_out = []
    for i, frame in enumerate(data_in):
        snr_in.append(calculate_snr(frame, data_mean0[i, :]))
        snr_out.append(calculate_snr(pred_out[i, :], data_mean0[i, :]))
    del i, frame, model_test

    snr_in = np.array(snr_in)
    snr_out = np.array(snr_out)
    snr_delta = snr_out - snr_in

    print(f"\nCalcuted SNR values from inference on validated datas")
    print(f"- SNR_in: {np.median(snr_in):.2f} (median) | {np.max(snr_in):.2f} (max) | {np.min(snr_in):.2f} (min)")
    print(f"- SNR_out: {np.median(snr_out):.2f} (median) | {np.max(snr_out):.2f} (max) | {np.min(snr_out):.2f} (min)")
    print(f"- SNR_inc: {np.median(snr_delta): .2f} (median)")

    # --- Saving data
    logsdir = trainhandler.get_saving_path()
    savemat(join(logsdir, 'results.mat'),
            {"frames_in": data_in,
             "frames_out": data_out,
             "frames_mean": data_mean,
             "frames_pred": pred_out,
             "feat": feat_out,
             "cluster": cluster_out,
             "config": config_train},
            do_compression=True,
            long_field_names=True)

    # --- Plotting
    results_training(
        path=logsdir, feat=feat_out,
        yin=data_in, ypred=pred_out, ymean=data_mean,
        cluster=cluster_out, snr=snr_train
    )
    plot_statistic_data(yclus, cluster_out, path2save=logsdir, cl_dict=dataset_dict)

    plt.show(block=False)
    plt.close("all")
