import numpy as np
from datetime import datetime
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src.processing_noise import frame_noise

# TODO: Rauschen mit SNR hier korrelieren
class DatasetDAE(Dataset):
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

def get_dataloaders(dataset: DatasetDAE, batch_size: int, validation_split: float, shuffle: bool) -> tuple[DataLoader, DataLoader]:
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

def generate_frames(no_frames: int, frame_in: np.ndarray, cluster_in: int, noise_lvl: [float, float]) -> tuple[np.ndarray, np.ndarray]:
    fs = 20e3
    new_cluster = cluster_in * np.ones(shape=(no_frames,), dtype=int)
    _, new_frame = frame_noise(no_frames, frame_in, noise_lvl, fs)
    return new_cluster, new_frame

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

def change_frame_size(frames_in: np.ndarray, sel_pos: list) -> np.ndarray:
    if (len(sel_pos) != 2):
        # Alle Werte übernehmen
        frames_out = frames_in
    else:
        # Fensterung der Frames
        frames_out = frames_in[:, sel_pos[0]:sel_pos[1]]

    return frames_out

def calculate_mean_waveform(frames_in: np.ndarray, frames_cluster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculating mean waveforms of spike waveforms"""
    NoCluster, NumCluster = np.unique(frames_cluster, return_counts=True)
    # NoCluster = NoCluster.tolist()
    SizeCluster = np.size(NoCluster)
    SizeFrame = frames_in.shape[1]

    frames_mean = np.zeros(shape=(SizeCluster, SizeFrame), dtype=int)
    cluster_snr = np.zeros(shape=(SizeCluster, 3), dtype=int)
    for idx0, val in enumerate(NoCluster):
        # --- Mean waveform
        indices = np.where(frames_cluster == val)
        frames_sel = frames_in[indices[0], :]
        mean = np.mean(frames_sel, axis=0, dtype=int)
        frames_mean[idx0, :] = mean

        # --- Calculating SNR
        snr0 = np.zeros(shape=(indices[0].size,), dtype=float)
        for i, frame in enumerate(frames_sel):
            snr0[i] = calculate_snr(frame, mean)

        cluster_snr[idx0, 0] = np.min(snr0)
        cluster_snr[idx0, 1] = np.mean(snr0)
        cluster_snr[idx0, 2] = np.max(snr0)

    return frames_mean, cluster_snr

def augmentation_data(frames_mean: np.ndarray, frames_cluster: np.ndarray, snr_cluster: np.ndarray, num_min_frames: int, run: bool) -> tuple[np.ndarray, np.ndarray]:
    """Tool for data augmentation of input spike frames"""
    frames_out = np.array([], dtype='float')
    cluster_out = np.array([], dtype='int')

    NoCluster, NumCluster = np.unique(frames_cluster, return_counts=True)
    # --- Adding artificial noise frames (Augmented Path)
    if run:
        noise_lvl = [-1000, -800]
        # noise_lvl = [snr_cluster[idx0, 0], snr_cluster[idx0, 2]]
        maxY = np.max(NumCluster)

        for idx0, val in enumerate(NumCluster):
            no_frames = num_min_frames + maxY - val
            (new_cluster, new_frame) = generate_frames(no_frames, frames_mean[idx0, :], NoCluster[idx0], noise_lvl)
            # Adding to output
            frames_out = new_frame if idx0 == 0 else np.append(frames_out, new_frame, axis=0)
            cluster_out = new_cluster if idx0 == 0 else np.append(cluster_out, new_cluster, axis=0)

    return frames_out, cluster_out

def generate_zero_frames(SizeFrame: int, num_frames: int, run: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generating zero frames with noise for data augmentation"""
    noise_lvl = [-100, -80]
    if not run:
        mean = np.array([], dtype='int')
        cluster = np.array([], dtype='int16')
        frames = np.array([], dtype='int16')
    else:
        mean = np.zeros(shape=(SizeFrame, ), dtype='int16')
        (cluster, frames) = generate_frames(num_frames, mean, 0, noise_lvl)

    return mean, cluster, frames

def prepare_dae_training(path: str, do_addnoise: bool, num_min_frames: int, excludeCluster: list, sel_pos: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Einlesen des Datensatzes inkl. Augmentierung (Kein Pre-Processing)"""

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

    frames_in = change_frame_size(frames_in, sel_pos)
    frames_mean, snr_mean = calculate_mean_waveform(frames_in, frames_cluster)
    new_frames, new_clusters = augmentation_data(frames_mean, frames_cluster, snr_mean, num_min_frames, do_addnoise)
    if do_addnoise:
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cluster = np.append(frames_cluster, new_clusters, axis=0)

    # --- Exclusion of falling clusters
    if (len(excludeCluster) == 0):
        frames_in = frames_in
        frames_cluster = frames_cluster
    else:
        for i, id in enumerate(excludeCluster):
            selX = np.where(frames_cluster != id)
            frames_in = frames_in[selX[0], :]
            frames_cluster = frames_cluster[selX]

    do_addzeros = True
    new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_min_frames, do_addzeros)
    if do_addzeros:
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cluster = np.append(1+frames_cluster, new_clusters, axis=0)
        frames_mean = np.vstack([new_mean, frames_mean])

    return frames_in, frames_cluster, frames_mean

def calculate_snr(yin: np.ndarray, ymean: np.ndarray):
    """Calculating the signal-to-noise ratio of the input signal compared to mean waveform"""
    A = np.sum(np.square(yin))
    B = np.sum(np.square(ymean - yin))
    outdB = 10 * np.log10(A/B)
    return outdB