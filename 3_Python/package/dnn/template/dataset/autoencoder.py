import numpy as np
from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset

from package.dnn.pytorch_dataclass import Config_Dataset
from package.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib, generate_zero_frames
from package.data_process.frame_normalization import DataNormalization
from package.data_process.frame_augmentation import augmentation_change_position, augmentation_reducing_samples


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


def prepare_training(settings: Config_Dataset, do_classification=False,
                     use_median_for_mean=True, add_noise_cluster=False,
                     mode_train_ae=0, noise_std=0.1, print_state=True) -> DatasetAE:
    """Preparing dataset incl. augmentation for spike-frame based training
    Args:
        settings:               Class for loading the data and do pre-processing
        do_classification:      Decision if output should be a classification
        path2model:             Path to already-trained autoencoder
        add_noise_cluster:      Adding the noise cluster to dataset
        use_median_for_mean:    Using median for calculating mean waveform (Boolean)
        mode_train_ae:          Mode for training the autoencoder (0: normal, 1: Denoising (mean), 2: Denoising (input))
        noise_std:              Std of noise distribution
        print_state:            Printing the state and results into Terminal
    Returns:
        Dataloader for training autoencoder-based classifier
    """
    print("... loading and processing the dataset")
    npzfile = loadmat(settings.get_path2data)
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten() if 'frames_cluster' in npzfile else npzfile["frames_cl"].flatten()
    frames_dict = None

    # --- Using cell_bib for clustering
    if settings.use_cell_library:
        frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(
            settings.get_path2data,
            settings.use_cell_library,
            frames_in,
            frames_cl
        )

    # --- PART: Reducing samples per cluster (if too large)
    if settings.reduce_samples_per_cluster_do:
        if print_state:
            print("... do data augmentation with reducing the samples per cluster")
        frames_in, frames_cl = augmentation_reducing_samples(
            frames_in, frames_cl,
            settings.reduce_samples_per_cluster_num,
            do_shuffle=False
        )

    # --- PART: Data Normalization
    if settings.normalization_do:
        if print_state:
            print(f"... do data normalization")
        data_class_frames_in = DataNormalization('minmax', mode=settings.normalization_method)
        frames_in = data_class_frames_in.normalize(frames_in)

    # --- PART: Mean waveform calculation and data augmentation
    if use_median_for_mean:
        frames_me = calculate_frame_median(frames_in, frames_cl)
    else:
        frames_me = calculate_frame_mean(frames_in, frames_cl)

    # --- PART: Exclusion of selected clusters
    if len(settings.exclude_cluster) == 0:
        frames_in = frames_in
        frames_cl = frames_cl
    else:
        for i, id in enumerate(settings.exclude_cluster):
            selX = np.where(frames_cl != id)
            frames_in = frames_in[selX[0], :]
            frames_cl = frames_cl[selX]

    # --- Generate dict with labeled names
    if frames_dict is None:
        frames_dict = list()
        for id in np.unique(frames_cl):
            frames_dict.append(f"Neuron #{id}")

    # --- PART: Calculate SNR if desired
    if settings.augmentation_do or add_noise_cluster:
        snr_mean = calculate_frame_snr(frames_in, frames_cl, frames_me)
    else:
        snr_mean = np.zeros(0, dtype=float)

    # --- PART: Data Augmentation
    if settings.augmentation_do and not settings.reduce_samples_per_cluster_do:
        if print_state:
            print("... do data augmentation")
        new_frames, new_clusters = augmentation_change_position(
            frames_in, frames_cl, snr_mean, settings.augmentation_num)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, new_clusters, axis=0)

    # --- PART: Generate and add noise cluster
    if add_noise_cluster:
        snr_range_zero = [np.median(snr_mean[:, 0]), np.median(snr_mean[:, 2])]
        info = np.unique(frames_cl, return_counts=True)
        num_cluster = np.max(info[0]) + 1
        num_frames = np.max(info[1])
        if print_state:
            print(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

        new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
        frames_me = np.vstack([frames_me, new_mean])

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
