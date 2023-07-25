import os.path
import glob
import matplotlib.pyplot as plt
import torch
import torchinfo
from scipy.io import savemat
import numpy as np

from src.metric import calculate_snr
from src_ai.dataset_preparation import prepare_training
from src_ai.pytorch_handler import training_pytorch
# from src_ai.dae_dataset import DatasetDAE as DatasetUsed, prepare_plotting, get_dataloaders
from src_ai.ae_dataset import DatasetAE as DatasetUsed, prepare_plotting, get_dataloaders
from src_ai.ae_topology import dnn_ae_v1 as ai_module
# from src_ai.dae_topology_embedded import dnn_dae_v1 as ai_module
import src_ai.plotting as plt_spaike


def loss_func(feat_1, feat_2):
    scaling = [5, 4]
    func0 = 1 - torch.nn.functional.cosine_similarity(feat_1, feat_2).mean()
    func1 = torch.nn.functional.mse_loss(feat_1, feat_2)

    out = scaling[0] * func0 + scaling[1] * func1
    return out


# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    path2data = 'data'
    index_folder = 'train'
    file_name = '2023-05-15_Dataset01_SimDaten_Martinez2009_Sorted'
    # file_name = '2023-06-30_Dataset03_SimDaten_Quiroga2020_Sorted'

    no_epochs = 2000
    batch_size = 64
    augment_do = False
    augment_num = 0
    noise_do = False
    excludeCluster = [1]
    sel_pos = []

    # --- Pre-Processing: Loading data and splitting into training and validation
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = prepare_training(
        path=path, do_augmentation=augment_do, num_new_frames=augment_num,
        excludeCluster=excludeCluster, sel_pos=sel_pos,
        do_zeroframes=noise_do
    )
    dataset = DatasetUsed(frames_in, frames_cluster, frames_mean)
    train_dl, valid_dl = get_dataloaders(
        dataset, batch_size=batch_size,
        validation_split=0.2,
        shuffle=True
    )

    # --- Pre-Processing: Create NN
    model = ai_module()
    model_name = model.out_modelname
    model_typ = model.out_modeltyp

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # loss_fn = loss_func

    torchinfo.summary(model)
    # --- Processing: Do Training
    trainhandler = training_pytorch(model_typ, model_name, index_folder)
    trainhandler.load_model(model, optimizer, loss_fn, no_epochs)
    trainhandler.load_data(train_dl, valid_dl)
    snr_train = trainhandler.do_training()
    logsdir = trainhandler.path2save

    # --- Post-Train-Processing: Getting data from validation set for plotting
    model_name_test = glob.glob(os.path.join(logsdir, 'model_*'))
    model_test = torch.load(model_name_test[0])

    data_in, data_out, cluster_out = prepare_plotting(valid_dl)
    model_in = torch.from_numpy(data_in)
    feat_out, pred_out = model_test(model_in)
    feat0 = feat_out.detach().numpy()
    ypred0 = pred_out.detach().numpy()

    # --- Post-Processing: SNR improvement
    snr_before = []
    snr_after = []
    for i, frame in enumerate(data_in):
        xin = frame
        fmean = data_out[i, :]
        fpred = ypred0[i, :]
        snr_before.append(calculate_snr(xin, fmean))
        snr_after.append(calculate_snr(fpred, fmean))

    snr_before = np.array(snr_before)
    snr_after = np.array(snr_after)

    print(
        f"... SNR before of: {np.mean(snr_before):.5f} (mean) - {np.max(snr_before):.5f} (max) - {np.min(snr_before):.5f} (min)")
    print(
        f"... SNR after of: {np.mean(snr_after):.5f} (mean) - {np.max(snr_after):.5f} (max) - {np.min(snr_after):.5f} (min)")

    # --- Saving data
    matdata = {"frames_in": data_in, "frames_out": data_out, "frames_pred": ypred0, "feat": feat0,
               "cluster": cluster_out}
    filename = 'results.mat'
    savemat(os.path.join(logsdir, filename), matdata)

    # --- Plotting
    plt_spaike.results_training(
        path=logsdir, yin=data_in,
        feat=feat0, ypred=ypred0,
        cluster=cluster_out,
        snr=snr_train
    )
    plt.show(block=False)

    # Look data on TensorBoard -> open Terminal
    # Type in: tensorboard serve --logdir ./runs
