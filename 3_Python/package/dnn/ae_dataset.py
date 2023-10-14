import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, frames: np.ndarray, index: np.ndarray,
                 mean_frame: np.ndarray,
                 mode_train=0):
        self.frames_orig = np.array(frames, dtype=np.float32)
        self.frames_noise = np.array(frames, dtype=np.float32)
        self.frames_mean = np.array(mean_frame, dtype=np.float32)
        self.cluster = index

        self.mode_train = mode_train
        if mode_train == 1:
            self.data_type = "Denoising Autoencoder (mean)"
        elif mode_train == 2:
            self.data_type = "Denoising Autoencoder (Add noise)"
        else:
            self.data_type = "Autoencoder"

    def __len__(self):
        return self.cluster.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.cluster[idx]
        frame_mean = self.frames_mean[cluster_id, :]

        if self.mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.frames_orig[idx, :]
            frame_out = self.frames_mean[cluster_id, :]
        elif self.mode_train == 2:
            # Denoising Autoencoder Training with adding noise on input
            frame_in = self.frames_noise[idx, :]
            frame_out = self.frames_orig[idx, :]
        else:
            # Normal Autoencoder Training
            frame_in = self.frames_orig[idx, :]
            frame_out = self.frames_orig[idx, :]

        return {'in': frame_in, 'out': frame_out, 'cluster': cluster_id, 'mean': frame_mean}


def prepare_plotting(data_plot: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Getting data from DataLoader for Plotting Results"""
    din = []
    dout = []
    did = []
    dmean = []
    for i, vdata in enumerate(data_plot):
        din0 = vdata['in']
        dout0 = vdata['out']
        dmean0 = vdata['mean']
        did0 = vdata['cluster']

        din = din0 if i == 0 else np.append(din, din0, axis=0)
        dout = dout0 if i == 0 else np.append(dout, dout0, axis=0)
        dmean = dmean0 if i == 0 else np.append(dmean, dmean0, axis=0)
        did = did0 if i == 0 else np.append(did, did0)

    return din, dout, did, dmean


def get_dataloaders(dataset: DatasetAE, batch_size=64,
                    validation_split=0.2, shuffle=False
                    ) -> tuple[DataLoader, DataLoader]:
    """Generate datasets for training and validation from input dataset"""
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
