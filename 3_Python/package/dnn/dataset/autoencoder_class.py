from os.path import join
from glob import glob
from scipy.io import loadmat
from torch import is_tensor, load, from_numpy
from torch.utils.data import Dataset

from package.dnn.pytorch_handler import Config_Dataset
from package.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.data_process.frame_preprocessing import change_frame_size, reconfigure_cluster_with_cell_lib, generate_zero_frames
from package.data_process.frame_normalization import DataNormalization
from package.data_process.frame_augmentation import *


class DatasetAE_Class(Dataset):
    """Dataset Preparation for training autoencoder-based classifications"""

    def __init__(self, frames_raw: np.ndarray, frames_feat: np.ndarray,
                 cluster_id: np.ndarray, frames_cluster_me: np.ndarray,
                 cluster_dict=None):
        self.size_output = 4
        # --- Input Parameters
        self.__frames_raw = np.array(frames_raw, dtype=np.float32)
        self.__frames_feat = np.array(frames_feat, dtype=np.float32)
        self.__cluster_id = np.array(cluster_id, dtype=np.uint8)
        self.frames_me = np.array(frames_cluster_me, dtype=np.float32)
        # --- Parameters for Confusion Matrix for Classification
        self.cluster_name_available = isinstance(cluster_dict, list)
        self.frame_dict = cluster_dict
        self.data_type = "Autoencoder-based Classification"

    def __len__(self):
        return self.__cluster_id.shape[0]

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        return {'in': self.__frames_feat[idx, :],
                'out': self.__cluster_id[idx]}


def prepare_training(settings: Config_Dataset, path2model: str,
                     use_cell_bib=False, mode_classes=2,
                     use_median_for_mean=True) -> DatasetAE_Class:
    """Preparing dataset incl. augmentation for spike-frame based training"""
    print("... loading and processing the dataset")
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
        print("... reducing the samples per cluster (for pre-training on dedicated hardware)")
        frames_in, frames_cl = augmentation_reducing_samples(frames_in, frames_cl,
                                                             settings.data_num_samples_per_cluster,
                                                             do_shuffle=False)

    # --- PART: Data Normalization
    if settings.data_do_normalization:
        print(f"... do data normalization")
        data_class_frames_in = DataNormalization(device=settings.data_normalization_mode,
                                                 method=settings.data_normalization_method,
                                                 mode=settings.data_normalization_setting)
        frames_in = data_class_frames_in.normalize(frames_in)

    # --- PART: Mean waveform calculation and data augmentation
    frames_in = change_frame_size(frames_in, settings.data_sel_pos)
    if use_median_for_mean:
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

    # --- Generate dict with labeled names
    if frames_dict is None:
        frames_dict = list()
        for id in np.unique(frames_cl):
            frames_dict.append(f"Neuron #{id}")

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
    print("... for training are", frames_in.shape[0], "frames with each", frames_in.shape[1], "points available")
    print(f"... used data points for training: in total {check[0].size} classes with {np.sum(check[1])} samples")
    for idx, id in enumerate(check[0]):
        addon = f'' if not isinstance(frames_dict, list | np.ndarray) else f' ({frames_dict[id]})'
        print(f"\tclass {id}{addon} --> {check[1][idx]} samples")

    return DatasetAE_Class(frames_raw=frames_in, frames_feat=frames_feat, cluster_id=frames_cl,
                           frames_cluster_me=frames_me, cluster_dict=frames_dict)
