import os

from scipy.io import loadmat
from torch import is_tensor
from torch.utils.data import Dataset

from package.dnn.pytorch_handler import ConfigDataset
from package.dnn.data_preprocessing_frames import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.dnn.data_preprocessing_frames import change_frame_size, reconfigure_cluster_with_cell_lib, generate_zero_frames, DataNormalization
from package.dnn.data_augmentation_frames import *


class DatasetAE(Dataset):
    """Dataset Preparator for training Autoencoder"""
    def __init__(self, frames_raw: np.ndarray, cluster_id: np.ndarray,
                 frames_cluster_me: np.ndarray, cluster_dict=None,
                 noise_std=0.1, do_classification=False, mode_train=0):

        # --- Input Parameters
        self.__frames_orig = np.array(frames_raw, dtype=np.float32)
        self.__frames_size = frames_raw.shape[1]
        self.__cluster_id = np.array(cluster_id, dtype=np.uint8)
        self.frames_me = np.array(frames_cluster_me, dtype=np.float32)
        # --- Parameters for Denoising Autoencoder
        self.__frames_noise_std = noise_std
        self.__do_classification = do_classification
        # --- Parameters for Confusion Matrix for Classification
        self.cluster_name_available = isinstance(cluster_dict, list)
        self.frame_dict = cluster_dict

        self.mode_train = mode_train
        if mode_train == 1:
            self.data_type = "Denoising Autoencoder (mean)"
        elif mode_train == 2:
            self.data_type = "Denoising Autoencoder (Add random noise)"
        elif mode_train == 3:
            self.data_type = "Denoising Autoencoder (Add gaussian noise)"
        else:
            self.data_type = "Autoencoder"

        if do_classification:
            self.data_type += " for Classification"

    def __len__(self):
        return self.__cluster_id.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        cluster_id = self.__cluster_id[idx]
        if self.mode_train == 1:
            # Denoising Autoencoder Training with mean
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.frames_me[cluster_id, :] if not self.__do_classification else cluster_id
        elif self.mode_train == 2:
            # Denoising Autoencoder Training with adding random noise on input
            frame_in = self.__frames_orig[idx, :] + np.array(self.__frames_noise_std * np.random.randn(self.__frames_size), dtype=np.float32)
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id

        elif self.mode_train == 3:
            # Denoising Autoencoder Training with adding gaussian noise on input
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id
            frame_in = self.__frames_orig[idx, :] + np.array(self.__frames_noise_std * np.random.normal(size=self.__frames_size), dtype=np.float32)
        else:
            # Normal Autoencoder Training
            frame_in = self.__frames_orig[idx, :]
            frame_out = self.__frames_orig[idx, :] if not self.__do_classification else cluster_id

        return {'in': frame_in, 'out': frame_out, 'class': cluster_id,
                'mean': self.frames_me[cluster_id, :]}


def prepare_training(settings: ConfigDataset,
                     use_cell_bib=False, mode_classes=2,
                     use_median=True,
                     mode_train_ae=0, do_classification=False,
                     noise_std=0.1) -> DatasetAE:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
    # Construct the full path
    full_path = settings.get_path2data()

    print(f"Constructed Path: {full_path}")
    print(f"Path Exists: {os.path.exists(full_path)}")
    # if os.path.exists(full_path) == False:
    # stop TRAINING
    try:
        data = np.load(full_path)
        print("File loaded successfully!")
    except FileNotFoundError as e:
        print(f"Error: {e}")

    npzfile = loadmat(settings.get_path2data())
    frames_in = npzfile["frames_in"]
    frames_cl = npzfile["frames_cluster"].flatten() if 'frames_cluster' in npzfile else npzfile["frames_cl"].flatten()
    frames_dict = None

    # --- Using cell_bib for clustering
    if use_cell_bib:
        frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(settings.get_path2data(),
                                                                              mode_classes, frames_in, frames_cl)

    # --- PART: Reducing samples per cluster (if too large)
    if settings.data_do_reduce_samples_per_cluster:
        print("... do data augmentation with reducing the samples per cluster")
        frames_in, frames_cl = augmentation_reducing_samples(frames_in, frames_cl,
                                                             settings.data_num_samples_per_cluster, False)

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        if settings.data_normalization_setting == 'bipolar':
            do_bipolar = True
            do_global = False
        elif settings.data_normalization_setting == 'global':
            do_bipolar = False
            do_global = True
        elif settings.data_normalization_setting == 'combined':
            do_bipolar = True
            do_global = True
        else:
            do_bipolar = False
            do_global = False
        print(f"... do data normalization")
        data_class_frames_in = DataNormalization(mode=settings.data_normalization_mode,
                                                 method=settings.data_normalization_method,
                                                 do_bipolar=do_bipolar, do_global=do_global)
        frames_in = data_class_frames_in.normalize(frames_in)

    # --- PART: Mean waveform calculation and data augmentation
    frames_in = change_frame_size(frames_in, settings.data_sel_pos)
    if use_median:
        frames_me = calculate_frame_median(frames_in, frames_cl)
    else:
        frames_me = calculate_frame_mean(frames_in, frames_cl)

    # --- PART: Exclusion of selected clusters
    if len(settings.data_exclude_cluster) == 0:
        frames_in = frames_in
        frames_cl = frames_cl
    else:
        for i, id in enumerate(settings.data_exclude_cluster):
            selX = np.where(frames_cl != id)
            frames_in = frames_in[selX[0], :]
            frames_cl = frames_cl[selX]

    # --- PART: Calculate SNR if desired
    if settings.data_do_augmentation or settings.data_do_addnoise_cluster:
        snr_mean = calculate_frame_snr(frames_in, frames_cl, frames_me)
    else:
        snr_mean = np.zeros(0, dtype=float)

    # --- PART: Data Augmentation
    if settings.data_do_augmentation and not settings.data_do_reduce_samples_per_cluster:
        print("... do data augmentation")
        # new_frames, new_clusters = augmentation_mean_waveform(
        # frames_me, frames_cl, snr_mean, settings.data_num_augmentation)
        new_frames, new_clusters = augmentation_change_position(
            frames_in, frames_cl, snr_mean, settings.data_num_augmentation)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, new_clusters, axis=0)

    # --- PART: Generate and add noise cluster
    if settings.data_do_addnoise_cluster:
        snr_range_zero = [np.median(snr_mean[:, 0]), np.median(snr_mean[:, 2])]
        info = np.unique(frames_cl, return_counts=True)
        num_cluster = np.max(info[0]) + 1
        num_frames = np.max(info[1])
        print(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

        new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
        frames_me = np.vstack([frames_me, new_mean])

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    print(f"... for training are {frames_in.shape[0]} frames with each {frames_in.shape[1]} points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if not isinstance(frames_dict, list | np.ndarray) else f' ({frames_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE(frames_raw=frames_in, cluster_id=frames_cl, frames_cluster_me=frames_me,
                     mode_train=mode_train_ae, do_classification=do_classification,
                     noise_std=noise_std)
