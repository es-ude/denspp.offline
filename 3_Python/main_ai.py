import os.path
import matplotlib.pyplot as plt
import torch, torchinfo
from scipy.io import savemat
import numpy as np

from src_ai.pytorch_handler import do_training
from src_ai.dae_dataset import Dataset, get_dataloaders, prepare_dae_training, prepare_dae_plotting, calculate_snr
import src_ai.dae_topology as nnModules
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
    path2data = 'src_ai/data/Martinez_2009'
    file_name = '2023-04-17_Dataset01_SimDaten_Martinez2009_Sorted.mat'
    no_epochs = 1000
    addnoise_do = True
    addnoise_num = 2000
    excludeCluster = [1]
    sel_pos = [] #[1, 28]

    model = nnModules.dnn_dae_v1()
    model_name = "dnn_dae_v1"

    # --- Getting the data for training and validation
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = prepare_dae_training(
        path, addnoise_do, addnoise_num, excludeCluster, sel_pos
    )
    # pltSpAIke.test_plot(frames_in, frames_cluster)
    dataset = Dataset(frames_in, frames_cluster, frames_mean)
    train_dl, validation_dl = get_dataloaders(
        dataset, batch_size=256,
        validation_split=0.2,
        shuffle=True
    )
    torchinfo.summary(model)

    # --- Prepare Training
    loss_fn = torch.nn.MSELoss()
    # loss_fn = loss_func
    optimizer = torch.optim.Adam(model.parameters())

    logsdir, snr_plot = do_training(
        model, train_dl, validation_dl,
        optimizer, loss_fn,
        no_epochs, model_name
    )

    # --- Plotting some results with validation
    data_in, data_out, cluster_out = prepare_dae_plotting(validation_dl)
    feat_out, pred_out = model(torch.from_numpy(data_in))
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
        snr = snr_plot
    )
    plt.show(block = False)

    # Look data on TensorBoard -> open Terminal
    # Type in: tensorboard serve --logdir ./runs


