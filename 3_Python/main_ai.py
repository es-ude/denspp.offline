import os.path, shutil

import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchinfo

import src.plotting as pltSpAIke
from src_ai.processing_noise import generate_noiseframe
from src_ai.processing_data import prepare_data_ae_training

from scipy.io import savemat, loadmat
from src_ai.dataset import Dataset, get_dataloaders
import src_ai.nnpy_architecture as nnModules

def get_data(path: str, do_addnoise: bool, NoFramesNoise: int, excludeCluster: list, sel_pos: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Setzen des Ignorier-Clusters in Line 89
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- Data loading
    if file_name[-3:] == "npz":
        # --- NPZ reading file
        npzfile = np.load(path)
        frames_in = npzfile['arr_0']
        frames_cluster = npzfile['arr_2']
    else:
        # --- MATLAB reading file
        npzfile = loadmat(path)
        frames_in = npzfile["frames_in"]
        frames_cluster = npzfile["frames_cluster"].flatten()

    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    NoCluster = np.unique(frames_cluster).tolist()
    SizeCluster = np.size(NoCluster)

    # --- Calculation of the mean waveform
    SizeFrame = frames_in.shape[1]
    if(len(sel_pos) != 2):
        # Alle Werte übernehmen
        frames_in = frames_in
    else:
        # Fensterung der Frames
        SizeFrame = sel_pos[1] - sel_pos[0]
        frames_in = frames_in[:, sel_pos[0]:sel_pos[1]]

    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame), dtype=int)
    idx0 = 0
    for idx in NoCluster:
        selY = np.where(frames_cluster == idx)
        frames_sel = frames_in[selY[0], :]
        mean = np.mean(frames_sel, axis=0, dtype=int)
        frames_mean[idx0, :] = mean
        idx0 += 1

    # --- Exclusion of falling clusters
    if (len(excludeCluster) == 0):
        frames_in = frames_in
        frames_cluster = frames_cluster
    else:
        for idx in excludeCluster:
            selX = np.where(frames_cluster != idx)
            frames_in = frames_in[selX[0], :]
            frames_cluster = frames_cluster[selX]

    return frames_in, frames_cluster, frames_mean


def do_training(model: nn.Module, training_loader, validation_loader, optimizer, loss_fn, epochs: int, model_name: str):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path_save = 'runs/fashion_trainer_{}'.format(timestamp)
    writer = SummaryWriter(path_save)
    epoch_number = 0
    best_vloss = 1_000_000.

    timestamp_start = datetime.now()
    timestamp_string = timestamp_start.strftime('%H:%M:%S.%f')
    print(f'\nTraining starts on: {timestamp_string}')
    for epoch in range(epochs):
        # Make sure gradient tracking is on, and do a pass over the data
        # --- Training
        model.train(True)
        avg_loss = train_one_epoch(
            model, training_loader,
            optimizer, loss_fn,
            epoch_number, writer
        )

        # --- Validation
        # We don't need gradients on to do reporting
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            data_in = vdata['frame']
            data_out = vdata['mean_frame']

            _, pred_out = model(data_in)
            vloss = loss_fn(pred_out, data_out)
            running_vloss += vloss

        avg_vloss = running_vloss / (i+1)
        print(f'... loss of epoch {epoch+1}/{epochs} [{(epoch+1)/epochs*100: .2f} %]: train = {avg_loss: .5f}, valid = {avg_vloss: .5f}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            'Training vs. Validation Loss',
            {'Training': avg_loss, 'Validation': avg_vloss},
            epoch_number + 1
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = os.path.join(path_save, 'model_{}'.format(epoch_number))
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    # --- Ausgabe nach Training
    timestamp_end = datetime.now()
    timestamp_string = timestamp_end.strftime('%H:%M:%S.%f')

    diff_time = timestamp_end - timestamp_start
    diff_string = diff_time
    print(f'Training ends on: {timestamp_string}')
    print(f'Training runs: {diff_string}')

    print(f'\nSave best model: {model_path}')
    path_model = os.path.join('models', model_name + "_" + timestamp)
    shutil.copy(model_path, path_model)

# TODO: Implement early break training modus
def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of iter(training_loader)
    # so that we can track the batch index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        data_in = data['frame']
        data_out = data['mean_frame']

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        _, pred_out = model(data_in)
        # Compute the loss and its gradients
        loss = loss_fn(pred_out, data_out)
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        no_batch_print = 100
        if i % no_batch_print == (no_batch_print-1):
            # loss per batch
            last_loss = running_loss / no_batch_print
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


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
    NoFramesNoise = 1
    excludeCluster = []
    sel_pos = [] #[7, 35]

    model = nnModules.dnn_dae_v1()
    model_name = "dnn_dae_v1"

    # --- Getting the data for training and validation
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = get_data(
        path, do_addnoise, NoFramesNoise,
        excludeCluster, sel_pos
    )
    dataset = Dataset(frames_in, frames_cluster, frames_mean)
    train_dl, validation_dl = get_dataloaders(
        dataset, batch_size=8,
        validation_split=0.1,
        shuffle=True
    )
    torchinfo.summary(model)

    # --- Prepare Training
    #loss_fn = torch.nn.MSELoss()
    loss_fn = loss_func
    optimizer = torch.optim.Adam(model.parameters())

    do_training(
        model,
        train_dl, validation_dl,
        optimizer, loss_fn,
        no_epochs, model_name
    )

    # --- Plotting some results with validation
    iteNo = 0
    for i, vdata in enumerate(validation_dl):
        if (iteNo == 0):
            data_in = vdata['frame']
            data_out = vdata['mean_frame']
        else:
            data_in = np.append(data_in, vdata['frame'], axis=0)
            data_out = np.append(data_out, vdata['mean_frame'], axis=0)
        iteNo += 1

    _, pred_out = model(torch.from_numpy(data_in))

    pltSpAIke.plot_frames(data_in, data_out, pred_out.detach().numpy())
    plt.show(block=True)

    # Look data on TensorBoard -> open Terminal
    # Type in: tensorboard serve --logdir ./runs


