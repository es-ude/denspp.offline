import numpy as np
import torch
from torch.utils.data import Dataset


class DatasetBAE(Dataset):
    def __init__(self, frames: np.ndarray, spk_type: np.ndarray, cluster: np.ndarray):
        self.__frames = np.array(frames, dtype=np.float32)
        self.__spk_type = np.array(spk_type, dtype=np.int)
        self.__cluster = np.array(cluster, dtype=np.int)
        # 0: noise                  -> Do nothing
        # 1: artefact               -> Do nothing
        # 2: background activity    -> Only Spike Tick
        # 3: Unit                   -> Start Spike Sorting
        self.data_type = 'Background Activity Rejector'

    def __len__(self):
        return self.__spk_type.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        frame = self.__frames[idx, :]
        spk_type = self.__spk_type[idx]
        cluster = self.__cluster[idx]
        return {'frame': frame, 'type': spk_type, 'cluster': cluster}
