import numpy as np
from copy import deepcopy
from logging import getLogger, Logger
from denspp.offline.dnn import DatasetFromFile, SettingsDataset
from denspp.offline.dnn.model_library import CellLibrary
from denspp.offline.metric import calculate_snr_cluster
from denspp.offline.preprocessing import DataNormalization
from denspp.offline.data_call.call_cellbib import CellSelector
from denspp.offline.data_augmentation import (
    augmentation_changing_position,
    augmentation_reducing_samples,
    generate_zero_frames,
    calculate_frame_mean
)


class DataProcessor:
    _logger: Logger
    _settings: SettingsDataset

    def __init__(self, settings: SettingsDataset) -> None:
        """Class for pre-processing different types of datasets
        :param settings:    Settings for pre-processing the dataset
        :return:            None
        """
        self._logger = getLogger(__name__)
        self._settings = settings

    def exclude_cluster_from_dataset(self, dataset: DatasetFromFile) -> DatasetFromFile:
        """Excluding cluster IDs from dataset
        :param dataset:     Class with dataset content
        :return:            New dataset without cluster IDs
        """
        data_in = dataset.data.copy()
        data_cl = dataset.label.copy()
        data_na = dataset.dict.copy()
        data_me = dataset.mean.copy()
        for idx, remove_cl in enumerate(self._settings.exclude_cluster):
            pos = np.argwhere(data_cl == remove_cl).flatten()
            data_in = np.delete(data_in, pos, axis=0)
            data_cl = np.delete(data_cl, pos, axis=0)
            data_me = np.delete(data_me, remove_cl, axis=0)
            data_na.pop(remove_cl-idx)
        return DatasetFromFile(
            data=data_in,
            label=data_cl,
            dict=data_na,
            mean=data_me
        )

    def reconfigure_cluster_with_cell_lib(self, sel_mode_classes: int, dataset: DatasetFromFile) -> DatasetFromFile:
        """Function for reducing the samples for a given cell bib
        :param sel_mode_classes:    Number of classes to select for each cell bib (0= original, 1= Reduced, 2= Subgroup, 3= Subtype)
        :param dataset:             Old dataclass with original loaded dataset
        :return:                    New dataclass with reduced data content
        """
        cell_libs_handler = CellLibrary().get_registry()
        libs_class_overview = [lib.split("resort_")[-1] for lib in
                               cell_libs_handler.get_library_overview(do_print=False)]
        libs_use = [f'resort_{lib}' for lib in libs_class_overview if
                    lib in self._settings.data_type.lower()]
        if len(libs_use):
            cl_sampler = CellSelector(
                cell_merge=cell_libs_handler.build(libs_use[0]),
                mode=sel_mode_classes
            )
            cell_dict = cl_sampler.get_label_list()
            self._logger.info(f"... Cluster types before reconfiguration: {np.unique(dataset.label)}")

            cluster_new, data_new = cl_sampler.transform_data_into_new(dataset.label, dataset.data)
            self._logger.info(f"... Cluster types after reconfiguration: {np.unique(cluster_new)}")
            return DatasetFromFile(
                data=data_new,
                label=cluster_new,
                dict=cell_dict,
                mean=np.zeros(shape=(np.unique(cluster_new).size, *data_new.shape[1:]))
            )
        else:
            raise ValueError("No library found")

    def process_vision_datasets(self, data: DatasetFromFile) -> DatasetFromFile:
        """Function for processing pictures
        :param data:    Dataclass with dataset content
        :return:        Dataclass with DatasetFromFile with {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}
        """
        # --- Using cell library
        if self._settings.use_cell_sort_mode:
            data_used = self.reconfigure_cluster_with_cell_lib(
                sel_mode_classes=self._settings.use_cell_sort_mode,
                dataset=data
            )
            self._logger.info("... do resorting of labels")
        else:
            data_used = deepcopy(data)
        # --- Exclusion of selected clusters
        if len(self._settings.exclude_cluster):
            data_used = self.exclude_cluster_from_dataset(data_used)
            self._logger.info(f"... class reduction done to {np.unique(data_used.label).size} classes")
        # --- Normalization
        if self._settings.normalization_do:
            data_in = data_used.data / 255.
            self._logger.info("... do data normalization on input")
            data_used = DatasetFromFile(
                data=data_in,
                label=data_used.label,
                dict=data_used.dict,
                mean=data_used.mean,
            )
        # --- Data Augmentation
        if self._settings.augmentation_do:
            raise NotImplementedError("No augmentation method is implemented - Please disable flag!")
        if self._settings.reduce_samples_per_cluster_do:
            raise NotImplementedError(f"No reducing samples technique is implemented - Please disable flag!")
        # --- Return
        return data_used

    def process_timeseries_datasets(self, data: DatasetFromFile, add_noise_cluster: bool = False) -> DatasetFromFile:
        """Function for processing frames extracted from time series data
        :param data:                    Dataclass with dataset content
        :param add_noise_cluster:       Adding the noise cluster to dataset
        :return:                        Dataclass with DatasetFromFile with {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}
        """
        # --- Using cell_bib for clustering
        if self._settings.use_cell_sort_mode:
            data_used = self.reconfigure_cluster_with_cell_lib(
                sel_mode_classes=self._settings.use_cell_sort_mode,
                dataset=data
            )
            self._logger.info("... do resorting of labels")
        else:
            data_used = deepcopy(data)
        # --- Exclusion of selected clusters
        if len(self._settings.exclude_cluster):
            data_used = self.exclude_cluster_from_dataset(data_used)
            self._logger.info(f"... class reduction done to {np.unique(data_used.label).size} classes")
        # --- PART: Reducing samples per cluster (if too large)
        if self._settings.reduce_samples_per_cluster_do:
            frames_in, frames_cl = augmentation_reducing_samples(
                frames_in=data_used.data,
                frames_cl=data_used.label,
                num_frames=self._settings.reduce_samples_per_cluster_num,
                do_shuffle=True
            )
            data_used = DatasetFromFile(
                data=frames_in,
                label=frames_cl,
                dict=data_used.dict,
                mean=data_used.mean,
            )
            self._logger.info("... do data augmentation with reducing the samples per cluster")
        # --- PART: Data Normalization
        if self._settings.normalization_do:
            frames_in = DataNormalization(
                method=self._settings.normalization_method,
                do_global_scaling=True,
                peak_mode=0
            ).normalize(data_used.data)
            data_used = DatasetFromFile(
                data=frames_in,
                label=data_used.label,
                dict=data_used.dict,
                mean=data_used.mean,
            )
            self._logger.info(f"... do data normalization")
        # --- PART: Data Augmentation
        if self._settings.augmentation_do and not self._settings.reduce_samples_per_cluster_do:
            frames_in, frames_cl = augmentation_changing_position(
                frames_in=data_used.data,
                frames_cl=data_used.label,
                num_min_frames=self._settings.augmentation_num
            )
            data_used = DatasetFromFile(
                data=frames_in,
                label=frames_cl,
                dict=data_used.dict,
                mean=data_used.mean,
            )
            self._logger.info("... do data augmentation")
        # --- PART: Calculate mean waveforms for each cluster
        if data_used.label.size == data_used.data.shape[0]:
            data_used = DatasetFromFile(
                data=data_used.data,
                label=data_used.label,
                dict=data_used.dict,
                mean=calculate_frame_mean(data_used.data, data_used.label)
            )
        # --- PART: Generate and add noise cluster
        if add_noise_cluster:
            snr_mean = calculate_snr_cluster(data_used.data, data_used.label, data_used.mean)
            snr_range_zero = [np.median(snr_mean[:, 0]), np.median(snr_mean[:, 2])]
            info = np.unique(data_used.label, return_counts=True)
            num_cluster = np.max(info[0]) + 1
            num_frames = np.max(info[1])

            new_frames, new_clusters, new_mean = generate_zero_frames(data_used.data.shape[1], num_frames, snr_range_zero)
            label = data_used.dict.copy()
            label.extend(["noise"])
            data_used = DatasetFromFile(
                data=np.append(data_used.data, new_frames, axis=0),
                label=np.append(data_used.label, num_cluster + new_clusters, axis=0),
                mean=np.vstack([data_used.mean, new_mean]),
                dict=label,
            )
            self._logger.info(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")
        return data_used