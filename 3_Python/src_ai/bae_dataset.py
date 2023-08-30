import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class DatasetBAE(Dataset):
    def __init__(self, frames: np.ndarray, spk_type: np.ndarray, cluster: np.ndarray):
        self.frames = np.array(frames, dtype=np.float32)
        self.spk_type = np.array(spk_type, dtype=np.int)
        self.cluster = np.array(cluster, dtype=np.int)
        # 0: noise                  -> Do nothing
        # 1: artefact               -> Do nothing
        # 2: background activity    -> Only Spike Tick
        # 3: Unit                   -> Start Spike Sorting
    def  __len__(self):
        return self.spk_type.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.frames[idx, :]
        spk_type = self.spk_type[idx]
        cluster = self.cluster[idx]
        return {'frame': frame, 'type': spk_type, 'cluster': cluster}