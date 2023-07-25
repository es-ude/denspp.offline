import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""

    def __init__(self, frames: np.ndarray, index: np.ndarray, mean_frame: np.ndarray):
        self.frames = np.array(frames, dtype=np.float32)
        self.index = index
        self.use_dae = True

    def __len__(self):
        return self.index.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.frames[idx, :]
        cluster_id = self.index[idx]
        return {'in': frame, 'out': frame, 'cluster': cluster_id}


def prepare_plotting(data_plot: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    din = []
    dout = []
    did = []
    for i, vdata in enumerate(data_plot):
        if i == 0:
            din = vdata['in']
            did = vdata['cluster']
        else:
            din = np.append(din, vdata['in'], axis=0)
            did = np.append(did, vdata['cluster'])

    dout = din

    return din, dout, did


def get_dataloaders(dataset: DatasetAE, batch_size: int, validation_split: float, shuffle: bool) -> tuple[
    DataLoader, DataLoader]:
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
