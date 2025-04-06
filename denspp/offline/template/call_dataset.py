import numpy as np
from os.path import exists, join
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, ControllerDataset
from denspp.offline.dnn.model_library import CellLibrary
from denspp.offline.data_call.owncloud_handler import OwnCloudDownloader
from denspp.offline.data_process.frame_preprocessing import calculate_frame_snr, calculate_frame_mean
from denspp.offline.data_process.frame_preprocessing import reconfigure_cluster_with_cell_lib, generate_zero_frames
from denspp.offline.data_process.frame_normalization import DataNormalization
from denspp.offline.data_process.frame_augmentation import augmentation_change_position, augmentation_reducing_samples


class DatasetLoader(ControllerDataset):
    _settings: SettingsDataset

    def __init__(self, settings: SettingsDataset) -> None:
        """Class for downloading (function name with '__get_xyz')
        and preparing (function name with '__prepare_xyz') custom-defined datasets to train deep learning models
        :param settings:  Object of class SettingsDataset for handling dataset used in DeepLearning"""
        super().__init__(settings)

    def __download_spike(self, dataset_name: str) -> None:
        if not exists(self._settings.get_path2data):
            oc_handler = OwnCloudDownloader(self._settings.get_path2folder_project)
            oc_handler.download_file(
                use_dataset=True,
                file_name=dataset_name,
                destination_download=join(self._settings.get_path2folder, dataset_name)
            )
            oc_handler.close()

    def __preprocess_spike(self, add_noise_cluster: bool=False) -> dict:
        """Function for processing neural spike frame events from dataset
            :param add_noise_cluster:      Adding the noise cluster to dataset
            :return:                       Dict with {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}
        """
        # --- Loading rawdata ['data'=frames, 'label'= label id, 'peak'=amplitude values, 'dict'=label names]
        rawdata = np.load(self._settings.get_path2data, allow_pickle=True).flatten()[0]
        frames_dict = rawdata['dict']
        frames_in = rawdata['data']
        frames_cl = rawdata['label']

        # --- Using cell_bib for clustering
        cell_libs_handler = CellLibrary().get_registry()
        libs_class_overview = [lib.split("resort_")[-1] for lib in cell_libs_handler.get_model_library_overview(do_print=False)]
        libs_use = [f'resort_{lib}' for lib in libs_class_overview if lib in self._settings.get_path2data.lower()]
        if len(libs_use):
            new_data = reconfigure_cluster_with_cell_lib(
                fn=cell_libs_handler.build_model(libs_use[0]),
                sel_mode_classes=self._settings.use_cell_sort_mode,
                frames_in=frames_in,
                frames_cl=frames_cl
            )
            frames_in = new_data['frame']
            frames_cl = new_data['cl']
            frames_dict = new_data['dict']

        # --- PART: Reducing samples per cluster (if too large)
        if self._settings.reduce_samples_per_cluster_do:
            print("... do data augmentation with reducing the samples per cluster")
            frames_in, frames_cl = augmentation_reducing_samples(
                frames_in=frames_in,
                frames_cl=frames_cl,
                num_frames=self._settings.reduce_samples_per_cluster_num,
                do_shuffle=False
            )

        # --- PART: Exclusion of selected clusters
        if not len(self._settings.exclude_cluster) == 0:
            for id in self._settings.exclude_cluster:
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
        if self._settings.normalization_do:
            print(f"... do data normalization")
            data_class_frames_in = DataNormalization(self._settings.normalization_method)
            frames_in = data_class_frames_in.normalize(frames_in)

        # --- PART: Mean waveform calculation and data augmentation
        frames_me = calculate_frame_mean(frames_in, frames_cl)

        # --- PART: Calculate SNR if desired
        if self._settings.augmentation_do or add_noise_cluster:
            snr_mean = calculate_frame_snr(frames_in, frames_cl, frames_me)
        else:
            snr_mean = np.zeros(0, dtype=float)

        # --- PART: Data Augmentation
        if self._settings.augmentation_do and not self._settings.reduce_samples_per_cluster_do:
            print("... do data augmentation")
            new_frames, new_clusters = augmentation_change_position(
                frames_in=frames_in,
                frames_cl=frames_cl,
                num_min_frames=self._settings.augmentation_num
            )
            frames_in = np.append(frames_in, new_frames, axis=0)
            frames_cl = np.append(frames_cl, new_clusters, axis=0)

        # --- PART: Generate and add noise cluster
        if add_noise_cluster:
            snr_range_zero = [np.median(snr_mean[:, 0]), np.median(snr_mean[:, 2])]
            info = np.unique(frames_cl, return_counts=True)
            num_cluster = np.max(info[0]) + 1
            num_frames = np.max(info[1])
            print(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

            new_mean, new_clusters, new_frames = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
            frames_in = np.append(frames_in, new_frames, axis=0)
            frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
            frames_me = np.vstack([frames_me, new_mean])

        return {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}

    def __get_martinez(self) -> None:
        self.__download_spike('2023-05-15_Dataset_Sim_Martinez2009_Sorted.npy')

    def __prepare_martinez(self) -> dict:
        return self.__preprocess_spike()

    def __get_quiroga(self) -> None:
        self.__download_spike('2023-06-30_Dataset_Sim_Quiroga2020_Sorted.npy')

    def __prepare_quiroga(self) -> dict:
        return self.__preprocess_spike()
