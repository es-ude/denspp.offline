import numpy as np
from datetime import datetime
from scipy.io import loadmat

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


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
        return {'frame': frame, 'mean_frame': mean_frame}

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

def prepare_dae_training(path: str, do_addnoise: bool, do_reducesize: bool, excludeCluster: list, sel_pos: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    NoCluster = np.unique(frames_cluster).tolist()
    SizeCluster = np.size(NoCluster)

    # --- Calculation of the mean waveform + reducing of frames
    max_frames = 630
    SizeFrame = frames_in.shape[1]
    if(len(sel_pos) != 2):
        # Alle Werte übernehmen
        frames_in = frames_in
    else:
        # Fensterung der Frames
        SizeFrame = sel_pos[1] - sel_pos[0]
        frames_in = frames_in[:, sel_pos[0]:sel_pos[1]]

    #TODO: Arrayverkleinerung einfügen
    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame), dtype=int)
    frames_sel_new = np.zeros(shape=(SizeCluster, max_frames), dtype=int)
    idx0 = 0
    for idx in NoCluster:
        indices = np.where(frames_cluster == idx)
        frames_sel = frames_in[indices[0], :]
        mean = np.mean(frames_sel, axis=0, dtype=int)
        frames_mean[idx0, :] = mean
        # --- Extract specific amount of frame
        if do_reducesize:
            np.random.shuffle(indices[0])
            if idx0 == 0:
                frames_sel_new[idx0, :] = indices[0][:max_frames]
            else:
                frames_sel_new = indices[0][:max_frames]

        # Increasing counter
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

def prepare_dae_plotting(data_plot) -> tuple[np.ndarray, np.ndarray]:
    din = []
    dout = []
    iteNo = 0
    for i, vdata in enumerate(data_plot):
        if (iteNo == 0):
            din = vdata['frame']
            dout = vdata['mean_frame']
        else:
            din = np.append(din, vdata['frame'], axis=0)
            dout = np.append(dout, vdata['mean_frame'], axis=0)
        iteNo += 1

    return din, dout

