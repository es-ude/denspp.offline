from dataclasses import dataclass
from typing import Any
from os import getcwd, makedirs
from os.path import join, abspath, exists
from torch import optim, nn
import numpy as np

from package.data_call.owncloud_handler import owncloudDownloader
from package.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from package.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib, generate_zero_frames
from package.data_process.frame_normalization import DataNormalization
from package.data_process.frame_augmentation import augmentation_change_position, augmentation_reducing_samples



@dataclass
class Config_PyTorch:
    """Class for handling the PyTorch training/inference pipeline"""
    model_name: str
    patience: int
    optimizer: str
    loss: str
    deterministic_do: bool
    deterministic_seed: int
    num_kfold: int
    num_epochs: int
    batch_size: int
    data_split_ratio: float
    data_do_shuffle: bool

    def get_loss_func(self) -> Any:
        """Getting the loss function"""
        match self.loss:
            case 'MSE':
                loss_func = nn.MSELoss()
            case 'Cross Entropy':
                loss_func = nn.CrossEntropyLoss()
            case _:
                raise NotImplementedError("Loss function unknown! - Please implement or check!")
        return loss_func

    def load_optimizer(self, model, learn_rate=0.1) -> Any:
        """Loading the optimizer function"""
        match self.optimizer:
            case 'Adam':
                optim_func = self.__set_optimizer_adam(model)
            case 'SGD':
                optim_func = self.__set_optimizer_sgd(model, learn_rate=learn_rate)
            case _:
                raise NotImplementedError("Optimizer function unknown! - Please implement or check!")
        return optim_func

    def __set_optimizer_adam(self, model):
        """Using the Adam Optimizer"""
        return optim.Adam(model.parameters())

    def __set_optimizer_sgd(self, model, learn_rate=0.1):
        """Using the SGD as Optimizer"""
        return optim.SGD(model.parameters(), lr=learn_rate)


DefaultSettingsTrainMSE = Config_PyTorch(
    model_name='dnn_ae_v1',
    patience=20,
    optimizer='Adam',
    loss='MSE',
    deterministic_do=False,
    deterministic_seed=42,
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2
)
DefaultSettingsTrainCE = Config_PyTorch(
    model_name='dnn_cl_v1',
    patience=20,
    optimizer='Adam',
    loss='Cross Entropy',
    num_kfold=1,
    num_epochs=10,
    batch_size=256,
    data_do_shuffle=True,
    data_split_ratio=0.2,
    deterministic_do=False,
    deterministic_seed=42
)


@dataclass
class Config_Dataset:
    """Class for handling preparation of dataset"""
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    use_cell_library: int
    # --- Data Augmentation
    augmentation_do: bool
    augmentation_num: int
    normalization_do: bool
    normalization_method: str
    reduce_samples_per_cluster_do: bool
    reduce_samples_per_cluster_num: int
    # --- Dataset Preparation
    exclude_cluster: list

    @property
    def get_path2data(self) -> str:
        """Getting the path name to the file"""
        return abspath(join(self.get_path2folder, self.data_file_name))

    @property
    def get_path2folder(self) -> str:
        """Getting the path name to the file"""
        if self.data_path == 'data':
            path = join(self.get_path2folder_data)
        elif not self.data_path == '':
            path = join(self.get_path2folder_project, self.data_path)
        else:
            path = join(self.data_path)
        return abspath(path)

    @property
    def get_path2folder_data(self) -> str:
        """Getting the default path of the data inside the Python Project"""
        return abspath(join(self.get_path2folder_project, 'data'))

    @property
    def get_path2folder_project(self, start_folder='3_Python') -> str:
        """Getting the default path of the Python Project"""
        return abspath(join(getcwd().split(start_folder)[0], start_folder))

    def print_overview_datasets(self, do_print=True) -> list:
        """"""
        oc_handler = owncloudDownloader(self.get_path2folder_project, use_dataset=True)
        list_datasets = oc_handler.get_overview_data()
        if do_print:
            print("\nNo local dataset is available. Enter the number of available datasets from remote:"
                  "\n==============================================================================")
            for idx, file in enumerate(list_datasets):
                print(f"\t{idx}: \t{file}")

        oc_handler.close()
        return list_datasets

    def load_dataset(self) -> dict:
        """Loading the dataset from defined data file"""
        self.__download_if_missing()
        dataset = np.load(self.get_path2data, allow_pickle=True).flatten()[0]
        return self.__process_spike_dataset(dataset)

    def __download_if_missing(self) -> None:
        """"""
        if self.data_file_name == '':
            list_datasets = self.print_overview_datasets(True)
            sel_data = input()
            self.data_file_name = list_datasets[int(sel_data)]
        else:
            list_datasets = self.print_overview_datasets(False)
            for file in list_datasets:
                if self.data_file_name.lower() in file.lower():
                    self.data_file_name = file
                    break

        if not exists(self.get_path2data):
            makedirs(self.get_path2folder, exist_ok=True)

            oc_handler = owncloudDownloader(self.get_path2folder_project, use_dataset=True)
            oc_handler.download_file(self.data_file_name, self.get_path2data)
            oc_handler.close()

    def __process_spike_dataset(self, rawdata: dict, use_median_for_mean: bool = True, print_state: bool = True,
                                add_noise_cluster: bool = False) -> dict:
        """
        Args:
            data:
            use_median_for_mean:    Using median for calculating mean waveform (Boolean)
            print_state:            Printing the state and results into Terminal
            add_noise_cluster:      Adding the noise cluster to dataset
        Return:
            Dict
        """
        if print_state:
            print("... loading and processing the dataset")

        # --- Loading data
        frames_dict = rawdata['dict']
        if 'peak' in rawdata.keys():
            # TODO: Generate extra function for it and include in dataset settings
            if 'rgc_mcs' in self.data_file_name.lower():
                ignore_samples = np.argwhere(rawdata['peak'] >= 200.0).flatten()
                ignore_samples = np.concatenate((ignore_samples, np.argwhere(rawdata['peak'] <= 40.0).flatten()), axis=0)
                ignore_samples = np.concatenate((ignore_samples, np.argwhere(np.argmin(rawdata['data'], 1) != 40).flatten()), axis=0)

                frames_in0 = np.delete(rawdata['data'][:, 24:64], ignore_samples, 0)
                frames_cl = np.delete(rawdata['label'], ignore_samples, 0)
                frames_pk = np.delete(rawdata['peak'], ignore_samples, 0)
            else:
                frames_in0 = rawdata['data']
                frames_cl = rawdata['label']
                frames_pk = rawdata['peak']

            scale = np.repeat(np.expand_dims(frames_pk, axis=-1), frames_in0.shape[-1], axis=-1) / np.abs(frames_in0.min())
            frames_in = frames_in0 * scale
        else:
            frames_in = rawdata['data']
            frames_cl = rawdata['label']

        # --- Using cell_bib for clustering
        if self.use_cell_library:
            frames_in, frames_cl, frames_dict = reconfigure_cluster_with_cell_lib(
                self.get_path2data,
                self.use_cell_library,
                frames_in,
                frames_cl
            )

        # --- PART: Reducing samples per cluster (if too large)
        if self.reduce_samples_per_cluster_do:
            if print_state:
                print("... do data augmentation with reducing the samples per cluster")
            frames_in, frames_cl = augmentation_reducing_samples(
                frames_in, frames_cl,
                self.reduce_samples_per_cluster_num,
                do_shuffle=False
            )

        # --- PART: Exclusion of selected clusters
        if not len(self.exclude_cluster) == 0:
            for id in self.exclude_cluster:
                selX = np.argwhere(frames_cl == id).flatten()
                frames_in = np.delete(frames_in, selX, 0)
                frames_cl = np.delete(frames_cl, selX, 0)
                if isinstance(frames_dict, list):
                    frames_dict.pop(id)

        # --- Generate dict with labeled names
        if isinstance(frames_dict, dict):
            frames_dict = list()
            for id in np.unique(frames_cl):
                frames_dict.append(f"Neuron #{id}")

        # --- PART: Data Normalization
        if self.normalization_do:
            if print_state:
                print(f"... do data normalization")
            data_class_frames_in = DataNormalization(self.normalization_method)
            frames_in = data_class_frames_in.normalize(frames_in)

        # --- PART: Mean waveform calculation and data augmentation
        if use_median_for_mean:
            frames_me = calculate_frame_median(frames_in, frames_cl)
        else:
            frames_me = calculate_frame_mean(frames_in, frames_cl)

        # --- PART: Calculate SNR if desired
        if self.augmentation_do or add_noise_cluster:
            snr_mean = calculate_frame_snr(frames_in, frames_cl, frames_me)
        else:
            snr_mean = np.zeros(0, dtype=float)

        # --- PART: Data Augmentation
        if self.augmentation_do and not self.reduce_samples_per_cluster_do:
            if print_state:
                print("... do data augmentation")
            new_frames, new_clusters = augmentation_change_position(
                frames_in, frames_cl, snr_mean, self.augmentation_num)
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

        return {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}


DefaultSettingsDataset = Config_Dataset(
    data_path='data',
    data_file_name='rgc_mcs',
    use_cell_library=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=True,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[4]
)


if __name__ == "__main__":
    och = owncloudDownloader()
    overview = och.get_overview_data()

    print(DefaultSettingsDataset.get_path2data)
    data = DefaultSettingsDataset.load_dataset()

    from package.data_merge.merge_dataset_rgc_onoff_fzj import plot_frames_rgc_onoff_60mea
    plot_frames_rgc_onoff_60mea(data, plot_show=True)
    print(".done")
