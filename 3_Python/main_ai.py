import os.path

import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import src.plotting as pltSpAIke
from src_ai.processing_noise import generate_noiseframe
from src_ai.processing_data import prepare_data_ae_training

from scipy.io import savemat, loadmat
from src_ai.nn_pytorch import NeuralNetwork
from src_ai.dataset import Dataset, get_dataloaders
from src_ai.nnpy_architecture import dnn_autoencoder
#from src_ai.nn_tensorflow import NeuralNetwork
#from src_ai.nntf_architecture import cnn_autoencoder as nn_network

np.random.seed(42)

def get_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # ----- Settings for AI -----

    do_addnoise = False
    NoFramesNoise = 1
    # Setzen des Ignorier-Clusters in Line 89

    # ------ Loading Data
    datumHeute = date.today()
    str_datum = datumHeute.strftime("%Y%m%d-%H%M%S")
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    data_type = False

    if file_name[-3:] == "npz":
        # --- NPZ reading file
        npzfile = np.load(path)
        frames_in = npzfile['arr_0']
        frames_cluster = npzfile['arr_2']
    else:
        # --- MATLAB reading file
        npzfile = loadmat(path)
        frames_in = npzfile["frames_in"]
        frames_cluster = npzfile["frames_cluster"]

    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")

    # --- Calculation of the mean waveform
    NoCluster = np.unique(frames_cluster)
    SizeCluster = np.size(NoCluster)
    SizeFrame = frames_in.shape
    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame[1]), dtype=int)
    for idx in NoCluster:
        selX = np.where(frames_cluster == idx)
        #print(selX)
        frames_sel = frames_in[selX[1], :]
        #print(frames_sel)
        frames_mean[idx - 1, :] = np.mean(frames_sel, axis=0, dtype=int)
    return frames_in, frames_cluster, frames_mean


def train(model: nn.Module, training_loader, validation_loader, optimizer, loss_fn, epochs):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    epoch_number = 0
    best_vloss = 1_000_000.
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(validation_loader):
            v_input_frame = vdata['frame']
            v_mean_waveform = vdata['mean_frame']
            encoded_features, v_denoised_waveform = model(v_input_frame)
            vloss = loss_fn(v_denoised_waveform, v_mean_waveform)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def train_one_epoch(model, training_loader, optimizer, loss_fn, epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        input_frames = data['frame']
        mean_frame = data['mean_frame']

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        # Make predictions for this batch
        encoded_features, output_frames = model(input_frames)
        # Compute the loss and its gradients
        loss = loss_fn(output_frames, mean_frame)
        loss.backward()
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            #print(np.unique(mean_frame.detach().numpy(), axis=0))
            pltSpAIke.plot_frames(mean_frame.detach().numpy(), output_frames.detach().numpy())
            plt.show(block=False)
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


# TODO: Implement early break training modus
if __name__ == "__main__":
    plt.close('all')
    print("\nTrain modules of spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    path2data = '/Users/leoburon/work/Sp-AI-ke/3_Python/src_ai/data/Martinez_2009'
    file_name = '2023-03-16_Martinez2009_Sorted.mat'
    path = os.path.join(path2data, file_name)
    frames_in, frames_cluster, frames_mean = get_data(path)
    dataset = Dataset(frames_in, frames_cluster, frames_mean)
    train_dl, validation_dl = get_dataloaders(dataset, batch_size=32, validation_split=0.1, shuffle=True)
    print("... datasets for training are available")

    model = dnn_autoencoder()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    train(model, train_dl, validation_dl,optimizer, loss_fn, 100)




