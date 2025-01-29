import numpy as np
from torch import is_tensor
from torch.utils.data import Dataset
from package.dnn.pytorch_config_data import ConfigDataset


class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, frames_raw: np.ndarray, cluster_id: np.ndarray,
                 frames_cluster_me: np.ndarray, cluster_dict=None,
                 noise_std=0.1, do_classification=False, mode_train=0):

        # --- Input Parameters
        self.__frames_orig = np.array(frames_raw, dtype=np.float32)
        self.__frames_size = frames_raw.shape[1]
        self.__cluster_id = np.array(cluster_id, dtype=np.uint8)
        self.__frames_me = np.array(frames_cluster_me, dtype=np.float32)
        # --- Parameters for Denoising Autoencoder
        self.__frames_noise_std = noise_std
        self.__do_classification = do_classification
        # --- Parameters for Confusion Matrix for Classification
        self.__labeled_dictionary = cluster_dict if isinstance(cluster_dict, list) else []
        self.__mode_train = mode_train

    def __len__(self):
        return self.__cluster_id.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__cluster_id[idx]
        if self.__mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.__frames_me[cluster_id, :] if not self.__do_classification else cluster_id
        elif self.__mode_train == 2:
            # Denoising Autoencoder Training with adding random noise on input
            frame_in = self.__frames_orig[idx, :] + np.array(self.__frames_noise_std * np.random.randn(self.__frames_size), dtype=np.float32)
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id
        elif self.__mode_train == 3:
            # Denoising Autoencoder Training with adding gaussian noise on input
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id
            frame_in = self.__frames_orig[idx, :] + np.array(self.__frames_noise_std * np.random.normal(size=self.__frames_size), dtype=np.float32)
        else:
            # Normal Autoencoder Training
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id

        return {'in': frame_in, 'out': frame_out, 'class': cluster_id,
                'mean': self.__frames_me[cluster_id, :]}

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
        match self.__mode_train:
            case 1:
                out = "Denoising Autoencoder (mean)"
            case 2:
                out = "Denoising Autoencoder (Add random noise)"
            case 3:
                out = "Denoising Autoencoder (Add gaussian noise)"
            case _:
                out = "Autoencoder"
        if self.__do_classification:
            out += " for Classification"
        return out


def prepare_training(settings: ConfigDataset, do_classification=False,
                     mode_train_ae=0, noise_std=0.1, print_state=True) -> DatasetAE:
    """Preparing dataset incl. augmentation for spike-frame based training
    Args:
        settings:               Class for loading the data and do pre-processing
        do_classification:      Decision if output should be a classification
        mode_train_ae:          Mode for training the autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input))
        noise_std:              Std of noise distribution
        print_state:            Printing the state and results into Terminal
    Returns:
        Dataloader for training autoencoder-based classifier
    """
    rawdata = settings.load_dataset()
    frames_in = rawdata['data']
    frames_cl = rawdata['label']
    frames_dict = rawdata['dict']
    frames_me = rawdata['mean']

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    if print_state:
        print(f"... for training are {frames_in.shape[0]} frames with each {frames_in.shape[1]} points available")
        print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
        for idx, id in enumerate(check[0]):
            addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[id]})'
            print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE(frames_raw=frames_in, cluster_id=frames_cl, frames_cluster_me=frames_me,
                     cluster_dict=frames_dict, mode_train=mode_train_ae, do_classification=do_classification,
                     noise_std=noise_std)
