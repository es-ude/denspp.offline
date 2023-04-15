import os.path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchinfo

from src.pytorch_handler import do_training
from src.dae_dataset import Dataset, get_dataloaders, prepare_dae_training, prepare_dae_plotting
import src.dae_topology as nnModules
import src.plotting as pltSpAIke


def loss_func(feat_1, feat_2):
    return (- torch.nn.functional.cosine_similarity(feat_1, feat_2).mean() + 1) + torch.nn.functional.mse_loss(feat_1, feat_2)

# --- Hauptprogramm
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    path2data = 'src_ai/data/Martinez_2009'
    file_name = '2023-03-16_Martinez2009_Sorted.mat'
    no_epochs = 100
    do_addnoise = False
    do_reducedata = True
    NoFramesNoise = 0
    excludeCluster = []
    sel_pos = [] #[7, 35]

    model = nnModules.dnn_dae_v1()
    model_name = "dnn_dae_v1"

    # --- Getting the data for training and validation
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = prepare_dae_training(
        path, do_addnoise, do_reducedata,
        excludeCluster, sel_pos
    )
    dataset = Dataset(frames_in, frames_cluster, frames_mean)
    train_dl, validation_dl = get_dataloaders(
        dataset, batch_size=32,
        validation_split=0.1,
        shuffle=True
    )
    torchinfo.summary(model)

    # --- Prepare Training
    loss_fn = torch.nn.MSELoss()
    #loss_fn = loss_func
    optimizer = torch.optim.Adam(model.parameters())

    do_training(
        model, train_dl, validation_dl,
        optimizer, loss_fn,
        no_epochs, model_name
    )

    # --- Plotting some results with validation
    data_in, data_out = prepare_dae_plotting(validation_dl)
    _, pred_out = model(torch.from_numpy(data_in))

    pltSpAIke.plot_frames(data_in, data_out, pred_out.detach().numpy())
    plt.show(block=True)

    # Look data on TensorBoard -> open Terminal
    # Type in: tensorboard serve --logdir ./runs


