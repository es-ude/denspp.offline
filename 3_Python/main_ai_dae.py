import os.path, glob
import matplotlib.pyplot as plt
import torch, torchinfo
from scipy.io import savemat
import numpy as np

from src_ai.pytorch_handler import training_pytorch
from src_ai.dae_dataset import DatasetDAE, get_dataloaders, prepare_dae_training, prepare_dae_plotting, calculate_snr
import src_ai.dae_topology as nnModules
# import src_ai.dae_topology_embedded as nnModules
import src_ai.plotting as pltSpAIke

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
    file_name = '2023-04-17_Dataset01_SimDaten_Martinez2009_Sorted.mat'

    no_epochs = 100
    batch_size = 256
    addnoise_do = True
    addnoise_num = 5000
    excludeCluster = [1]
    sel_pos = []

    # --- Getting the data for training and validation
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = prepare_dae_training(
        path=path, do_augmentation=addnoise_do, num_new_frames=addnoise_num,
        excludeCluster=excludeCluster, sel_pos=sel_pos,
        do_zeroframes=True
    )

    dataset = DatasetDAE(frames_in, frames_cluster, frames_mean)
    train_dl, valid_dl = get_dataloaders(
        dataset, batch_size=batch_size,
        validation_split=0.2,
        shuffle=True
    )

    # --- Prepare Training
    model = nnModules.dnn_dae_v1()
    model_name = "dnn_dae_v1"

    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss()
    # loss_fn = loss_func

    torchinfo.summary(model)
    trainhandler = training_pytorch("dae", model_name)
    trainhandler.load_model(model, optimizer, loss_fn, no_epochs)
    trainhandler.load_data(train_dl, valid_dl)
    snr_train = trainhandler.do_training()
    logsdir = trainhandler.path2save

    # --- Plotting some results with validation
    model_name_test = glob.glob(os.path.join(logsdir, 'model_*'))
    model_test = torch.load(model_name_test[0])

    data_in, data_out, cluster_out = prepare_dae_plotting(valid_dl)
    model_in = torch.from_numpy(data_in)
    feat_out, pred_out = model_test(model_in)
    feat0 = feat_out.detach().numpy()
    ypred0 = pred_out.detach().numpy()

    # --- Increased SNR
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

    print(f"... SNR before of: {np.mean(snr_before):.5f} (mean) - {np.max(snr_before):.5f} (max) - {np.min(snr_before):.5f} (min)")
    print(f"... SNR after of: {np.mean(snr_after):.5f} (mean) - {np.max(snr_after):.5f} (max) - {np.min(snr_after):.5f} (min)")

    # --- Saving data
    matdata = {"frames_in": data_in, "frames_out": data_out, "frames_pred": ypred0, "feat": feat0, "cluster": cluster_out}
    filename = 'results.mat'
    savemat(os.path.join(logsdir, filename), matdata)

    # --- Plotting
    pltSpAIke.results_training(
        path=logsdir, yin=data_in,
        feat=feat0, ypred=ypred0,
        cluster = cluster_out,
        snr = snr_train
    )
    plt.show(block = False)

    # Look data on TensorBoard -> open Terminal
    # Type in: tensorboard serve --logdir ./runs


