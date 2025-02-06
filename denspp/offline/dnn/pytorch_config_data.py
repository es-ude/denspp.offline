from dataclasses import dataclass
from os import makedirs
from os.path import join, abspath, exists, basename
import numpy as np
from torch import concat
from torchvision import datasets, transforms

from .model_library import CellLibrary
from denspp.offline.structure_builder import get_path_project_start
from denspp.offline.data_call.owncloud_handler import OwncloudDownloader
from denspp.offline.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean, calculate_frame_median
from denspp.offline.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib, generate_zero_frames
from denspp.offline.data_process.frame_normalization import DataNormalization
from denspp.offline.data_process.frame_augmentation import augmentation_change_position, augmentation_reducing_samples


@dataclass
class ConfigDataset:
    """Class for handling preparation of dataset"""
    # --- Settings of Datasets
    data_path: str
    data_file_name: str
    use_cell_sort_mode: int
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
        a = join(self.get_path2folder, self.data_file_name)
        return a

    @property
    def get_path2folder(self) -> str:
        """Getting the path name to the file"""
        if not self.data_path == '':
            path = join(self.get_path2folder_project, self.data_path)
        else:
            path = join(self.data_path)
        return abspath(path)

    @property
    def get_path2folder_project(self) -> str:
        """Getting the default path of the Python Project"""
        return get_path_project_start()

    def print_overview_datasets(self, do_print: bool=True) -> list:
        """"""
        oc_handler = OwncloudDownloader(self.get_path2folder_project, use_dataset=True)
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
        return self.__process_spike_dataset() if not self.data_file_name.lower() == 'mnist' else self.__process_mnist_dataset()

    def __download_if_missing(self) -> None:
        """Function for calling a dataset from remote"""
        makedirs(self.get_path2folder, exist_ok=True)

        if self.data_file_name.lower() == 'mnist':
            self.__download_mnist()
        elif self.data_file_name == '':
            list_datasets = self.print_overview_datasets(True)
            sel_data = input()
            self.data_file_name = list_datasets[int(sel_data)]
            self.__download_spike()
        else:
            list_datasets = self.print_overview_datasets(False)
            for file in list_datasets:
                if self.data_file_name.lower() in file.lower():
                    self.data_file_name = basename(file)
                    break
            self.__download_spike()

    def __download_spike(self) -> None:
        if not exists(self.get_path2data):
            oc_handler = OwncloudDownloader(self.get_path2folder_project, use_dataset=True)
            oc_handler.download_file(self.data_file_name, self.get_path2data)
            oc_handler.close()

    def __download_mnist(self) -> None:
        do_download = not exists(self.get_path2data)
        datasets.MNIST(self.data_path, train=True, download=do_download)
        datasets.MNIST(self.data_path, train=False, download=do_download)

    def __process_mnist_dataset(self) -> dict:
        """"""
        # --- Resampling of MNIST dataset
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        data_train = datasets.MNIST(self.data_path, train=True, download=False, transform=transform)
        data_valid = datasets.MNIST(self.data_path, train=False, download=False, transform=transform)

        data_raw = concat((data_train.data, data_valid.data), 0).numpy()
        data_label = concat((data_train.targets, data_valid.targets), 0).numpy()
        data_dict = data_train.classes
        return {'data': data_raw, 'label': data_label, 'dict': data_dict}

    def __process_spike_dataset(self, use_median_for_mean: bool = True, print_state: bool = True,
                                add_noise_cluster: bool = False) -> dict:
        """Function for processing neural spike frame events from dataset
        Args:

            use_median_for_mean:    Using median for calculating mean waveform (Boolean)
            print_state:            Printing the state and results into Terminal
            add_noise_cluster:      Adding the noise cluster to dataset
        Return:
            Dict with {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}
        """
        if print_state:
            print("... loading and processing the dataset")

        # --- Loading rawdata ['data'=frames, 'label'= label id, 'peak'=amplitude values, 'dict'=label names]
        rawdata = np.load(self.get_path2data, allow_pickle=True).flatten()[0]
        frames_dict = rawdata['dict']
        frames_in = rawdata['data']
        frames_cl = rawdata['label']

        # --- Using cell_bib for clustering
        cell_libs_handler = CellLibrary().get_registry()
        libs_class_overview = [lib.split("resort_")[-1] for lib in cell_libs_handler.get_model_library_overview(do_print=False)]
        libs_use = [f'resort_{lib}' for lib in libs_class_overview if lib in self.get_path2data.lower()]
        if len(libs_use):
            new_data = reconfigure_cluster_with_cell_lib(
                fn=cell_libs_handler.build_model(libs_use[0]),
                sel_mode_classes=self.use_cell_sort_mode,
                frames_in=frames_in,
                frames_cl=frames_cl
            )
            frames_in = new_data['frame']
            frames_cl = new_data['cl']
            frames_dict = new_data['dict']

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
                frames_in=frames_in,
                frames_cl=frames_cl,
                num_min_frames=self.augmentation_num
            )
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


DefaultSettingsDataset = ConfigDataset(
    data_path='data/datasets',
    data_file_name='',
    use_cell_sort_mode=0,
    augmentation_do=False,
    augmentation_num=0,
    normalization_do=False,
    normalization_method='minmax',
    reduce_samples_per_cluster_do=False,
    reduce_samples_per_cluster_num=0,
    exclude_cluster=[]
)



