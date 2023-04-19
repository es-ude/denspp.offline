import numpy as np
from datetime import datetime
from scipy.io import loadmat
import matplotlib.pyplot as plt

from src_ai.processing_noise import gen_noise_frame

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler

class Dataset(Dataset):
    def __init__(self, frames: np.ndarray, index: np.ndarray, mean_frame: np.ndarray):
        self.frames = np.array(frames, dtype=np.float32)
        self.index = index
        self.mean_frame = np.array(mean_frame, dtype=np.float32)

    def  __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.frames[idx, :]
        cluster_id = self.index[idx]
        mean_frame = self.mean_frame[cluster_id, :]
        return {'frame': frame, 'mean_frame': mean_frame, 'cluster': cluster_id}

def get_dataloaders(dataset: Dataset, batch_size: int, validation_split: float, shuffle: bool) -> tuple[DataLoader, DataLoader]:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=train_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=valid_sampler
    )

    return train_loader, validation_loader

def prepare_dae_training(path: str, do_addnoise: bool, num_min_frames: int, excludeCluster: list, sel_pos: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Setzen des Ignorier-Clusters in Line 89
    str_datum = datetime.now().strftime('%Y%m%d %H%M%S')
    print(f"Running on {str_datum}")
    print("... loading the datasets")

    # --- Data loading
    if path[-3:] == "npz":
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

    ## --- Doing pre-processing
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    #frames_in = scaler.fit_transform(frames_in)

    # --- Calculation of the mean waveform + reducing of frames size (opt.)
    NoCluster = np.unique(frames_cluster).tolist()
    SizeCluster = np.size(NoCluster)
    NumCluster = np.zeros(shape=(SizeCluster, ), dtype=int)

    SizeFrame = frames_in.shape[1]
    if(len(sel_pos) != 2):
        # Alle Werte übernehmen
        frames_in = frames_in
    else:
        # Fensterung der Frames
        SizeFrame = sel_pos[1] - sel_pos[0]
        frames_in = frames_in[:, sel_pos[0]:sel_pos[1]]

    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame), dtype=int)
    for idx0, val in enumerate(NoCluster):
        indices = np.where(frames_cluster == val)
        NumCluster[idx0] = indices[0].size
        frames_sel = frames_in[indices[0], :]
        mean = np.mean(frames_sel, axis=0, dtype=int)
        frames_mean[idx0, :] = mean

    # --- Calcuting SNR
    SNRCluster = np.zeros(shape=(SizeCluster, 3), dtype=int)
    for idx0, val in enumerate(NoCluster):
        snr0 = np.zeros(shape=(indices[0].size,), dtype=float)
        for i, frame in enumerate(frames_sel):
            snr0[i] = calculate_snr(frame, mean)

        SNRCluster[idx0, 0] = np.min(snr0)
        SNRCluster[idx0, 1] = np.mean(snr0)
        SNRCluster[idx0, 2] = np.max(snr0)

    # --- Adding artificial noise frames (Augmented Path)
    if do_addnoise:
        maxY = np.max(NumCluster)
        mode = 0
        for idx0, val in enumerate(NumCluster):
            if mode == 0:
                # Anreichern bis Grenze und neue Frames
                no_frames = num_min_frames + maxY - val
            else:
                # Nur neue Frames
                no_frames = num_min_frames

            new_cluster = NoCluster[idx0] * np.ones(shape=(no_frames, ), dtype=int)
            noise_lvl = [SNRCluster[idx0, 0], SNRCluster[idx0, 2]]
            _, new_frame = gen_noise_frame(no_frames, frames_mean[idx0, :], noise_lvl)

            #plt.figure()
            #print(NoCluster[idx0], no_frames, new_frame.size)
            #plt.plot(np.transpose(new_frame[1:, :]), color='k')
            #plt.plot(np.transpose(frames_mean[idx0, :]), color='r')
            #plt.show(block = True)

            frames_in = np.append(frames_in, new_frame, axis=0)
            frames_cluster = np.append(frames_cluster, new_cluster, axis=0)

    # --- Exclusion of falling clusters
    if (len(excludeCluster) == 0):
        frames_in = frames_in
        frames_cluster = frames_cluster
    else:
        for i, id in enumerate(excludeCluster):
            selX = np.where(frames_cluster != id)
            frames_in = frames_in[selX[0], :]
            frames_cluster = frames_cluster[selX]

    return frames_in, frames_cluster, frames_mean

def prepare_dae_plotting(data_plot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    din = []
    dout = []
    did = []
    for i, vdata in enumerate(data_plot):
        if i == 0:
            din = vdata['frame']
            dout = vdata['mean_frame']
            did = vdata['cluster']
        else:
            din = np.append(din, vdata['frame'], axis=0)
            dout = np.append(dout, vdata['mean_frame'], axis=0)
            did = np.append(did, vdata['cluster'])

    return din, dout, did

def calculate_snr(yin: np.ndarray, ymean: np.ndarray):
    A = np.sum(np.square(yin))
    B = np.sum(np.square(ymean - yin))
    outdB = 10 * np.log10(A/B)
    return outdB