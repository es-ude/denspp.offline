import numpy as np
from os.path import join
from glob import glob
from scipy.io import loadmat
from torch import is_tensor, load, from_numpy
from torch.utils.data import Dataset

from package.dnn.pytorch_dataclass import Config_Dataset
from package.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib, generate_zero_frames
from package.data_process.frame_normalization import DataNormalization
from package.data_process.frame_augmentation import augmentation_change_position, augmentation_reducing_samples
from package.fpga.transfer_data_verilog import settings_data


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
    def get_dictionary(self) -> list:
        """Getting the dictionary of labeled dataset"""
        return self.__labeled_dictionary

    @property
    def get_topology_type(self) -> str:
        """Getting the information of used Autoencoder topology"""
        return "Autoencoder-based Classification"


def prepare_training(settings: Config_Dataset, path2model: str,
                     add_noise_cluster=False, use_median_for_mean=True,
                     print_state=True) -> DatasetAE_Class:
    """Preparing dataset incl. augmentation for spike-frame based training
    Args:
        settings:               Class for loading the data and do pre-processing
        path2model:             Path to already-trained autoencoder
        add_noise_cluster:      Adding the noise cluster to dataset
        use_median_for_mean:    Using median for calculating mean waveform (Boolean)
        print_state:            Printing state and results into Terminal
    Returns:
        Dataloader for training autoencoder-based classifier
    """
    if print_state:
        print("... loading and processing the dataset")
    data = settings.load_dataset()
    frames_in = data['data']
    frames_cl = data['label']
    frames_dict = data['dict']

    # --- Using cell_bib for clustering
    if settings.use_cell_library:
        frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(settings.get_path2data,
                                                                              settings.use_cell_library,
                                                                              frames_in, frames_cl)

    # --- PART: Reducing samples per cluster (if too large)
    if settings.reduce_samples_per_cluster_do:
        if print_state:
            print("... reducing the samples per cluster (for pre-training on dedicated hardware)")
        frames_in, frames_cl = augmentation_reducing_samples(frames_in, frames_cl,
                                                             settings.reduce_samples_per_cluster_num,
                                                             do_shuffle=False)

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
        # new_frames, new_clusters = augmentation_mean_waveform(
        # frames_me, frames_cl, snr_mean, settings.data_num_augmentation)
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
            print(f"... adding a non-neural noise-cluster with index #{num_cluster} and with {num_frames} samples")

        new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
        frames_in = np.append(frames_in, new_frames, axis=0)
        frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
        frames_me = np.vstack([frames_me, new_mean])

    # --- PART: Calculating the features with given Autoencoder model
    overview_model = glob(join(path2model, '*.pth'))
    model_ae = load(overview_model[0])
    model_ae = model_ae.to("cpu")
    feat = model_ae(from_numpy(np.array(frames_in, dtype=np.float32)))[0]
    frames_feat = feat.detach().numpy()

    # --- Output
    check = np.unique(frames_cl, return_counts=True)
    if print_state:
        print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
        print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
        for idx, id in enumerate(check[0]):
            addon = f'' if len(frames_dict) == 0 else f' ({frames_dict[id]})'
            print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE_Class(frames_raw=frames_in, frames_feat=frames_feat, cluster_id=frames_cl,
                           frames_cluster_me=frames_me, cluster_dict=frames_dict)
