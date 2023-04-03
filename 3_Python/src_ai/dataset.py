import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class Dataset(Dataset):
    def __init__(self, frames:np.ndarray, index: np.ndarray, mean_frame: np.ndarray):
        self.frames = np.array(frames, dtype=np.float32)
        self.index = index
        self.mean_frame = np.array(mean_frame, dtype=np.float32)
    def  __len__(self):
        return self.index.shape[1]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.frames[idx, :]
        cluster_id = self.index[0, idx]
        mean_frame = self.mean_frame[cluster_id]
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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    return train_loader, validation_loader
