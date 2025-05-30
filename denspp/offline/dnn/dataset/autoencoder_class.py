import numpy as np
from os.path import join
from glob import glob
from torch import is_tensor, load, from_numpy
from torch.utils.data import Dataset
from denspp.offline.data_process.frame_preprocessing import calculate_frame_mean


class DatasetAE_Class(Dataset):
    def __init__(self, frames_raw: np.ndarray, frames_feat: np.ndarray,
                 cluster_id: np.ndarray, frames_cluster_me: np.ndarray,
                 cluster_dict=None):
        """Dataset Preparation for training autoencoder-based classifications"""
        # --- Input Parameters
        self.__frames_raw = np.array(frames_raw, dtype=np.float32)
        self.__frames_feat = np.array(frames_feat, dtype=np.float32)
        self.__cluster_id = np.array(cluster_id, dtype=np.uint8)
        self.__frames_me = np.array(frames_cluster_me, dtype=np.float32)

        # --- Parameters for Confusion Matrix for Classification
        self.__labeled_dictionary = cluster_dict if isinstance(cluster_dict, list) else []

    def __len__(self):
        return self.__cluster_id.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {'in': self.__frames_feat[idx, :],
                'out': self.__cluster_id[idx]}

    @property
    def get_mean_waveforms(self) -> np.ndarray:
        """Getting the mean waveforms of dataset"""
        return self.__frames_me

    @property
    def get_cluster_num(self) -> int:
        """"""
        return int(np.unique(self.__cluster_id).size)

    @property
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return "Autoencoder-based Classification"


def prepare_training(rawdata: dict, path2model: str, print_state: bool=True) -> DatasetAE_Class:
    """Preparing dataset incl. augmentation for spike-frame based training
    Args:
        rawdata:        Dict with raw data for training ['data', 'label', 'dict', 'mean']
        path2model:     Path to already-trained autoencoder
        print_state:    Printing state and results into Terminal
    Returns:
        Dataloader for training autoencoder-based classifier
    """
    frames_in = rawdata['data']
    frames_cl = rawdata['label']
    frames_dict = rawdata['dict']
    frames_me = rawdata['mean'] if 'mean' in rawdata.keys() else calculate_frame_mean(frames_in, frames_cl, False)

    # --- PART: Calculating the features with given Autoencoder model
    overview_model = glob(join(path2model, '*.pt'))
    model_ae = load(overview_model[0], weights_only=False)
    model_ae = model_ae.to("cpu")
    feat = model_ae(from_numpy(np.array(frames_in, dtype=np.float32)))[0]
    frames_feat = feat.detach().numpy()

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    if print_state:
        print("... for training are", frames_feat.shape[0], "frames with each", frames_feat.shape[1], "extracted features available")
        print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
        for idx, id in enumerate(check[0]):
            addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[idx]})'
            print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE_Class(
        frames_raw=frames_in,
        frames_feat=frames_feat,
        cluster_id=frames_cl,
        frames_cluster_me=frames_me,
        cluster_dict=frames_dict
    )
