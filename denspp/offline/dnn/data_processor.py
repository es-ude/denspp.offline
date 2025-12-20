import numpy as np
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
        """"""
        self._logger = getLogger(__name__)
        self._settings = settings

    def reconfigure_cluster_with_cell_lib(self, fn, sel_mode_classes: int, dataset: DatasetFromFile) -> DatasetFromFile:
        """Function for reducing the samples for a given cell bib
        :param fn:                  Class with new resorting dictionaries
        :param sel_mode_classes:    Number of classes to select for each cell bib (0= original, 1= Reduced, 2= Subgroup, 3= Subtype)
        :param dataset:             Old dataclass with originial loaded dataset
        :return:                    New dataclass with reduced data content
        """
        cl_sampler = CellSelector(fn, sel_mode_classes)
        cell_dict = cl_sampler.get_label_list()
        self._logger.info(f"... Cluster types before reconfiguration: {np.unique(dataset.label)}")

        cluster_new, data_new = cl_sampler.transform_data_into_new(dataset.label, dataset.data)
        self._logger.info(f"... Cluster types after reconfiguration: {np.unique(cluster_new)}")
        return DatasetFromFile(
            data=data_new,
            label=cluster_new,
            dict=cell_dict,
            mean=None
        )

    def process_vision_datasets(self, picture: np.ndarray, label: np.ndarray) -> DatasetFromFile:
        # --- Normalization
        if self._settings.normalization_do:
            picture = picture / 255.0
            self._logger.info("... do data normalization on input")

        # --- Exclusion of selected clusters
        if len(self._settings.exclude_cluster):
            for i, id in enumerate(self._settings.exclude_cluster):
                selX = np.where(label != id)
                picture = picture[selX[0], :]
                label = label[selX]
            self._logger.info(f"... class reduction done to {np.unique(label).size} classes")

        # --- Using cell library
        if self._settings.use_cell_sort_mode:
            raise NotImplementedError("No cell library for this case is available - Please disable flag!")

        # --- Data Augmentation
        if self._settings.augmentation_do:
            raise NotImplementedError("No augmentation method is implemented - Please disable flag!")

        if self._settings.reduce_samples_per_cluster_do:
            raise NotImplementedError(f"No reducing samples technique is implemented - Please disable flag!")
        return DatasetFromFile(
            data=picture,
            label=label,
            dict=[],
            mean=np.zeros(shape=(10, 28, 28))
        )

    def process_timeseries_datasets(self, data: DatasetFromFile, add_noise_cluster: bool = False) -> DatasetFromFile:
        """Function for processing neural spike frame events from dataset
            :param data: Dataset content
            :param add_noise_cluster:      Adding the noise cluster to dataset
            :return:                       Dict with {'data': frames_in, 'label': frames_cl, 'dict': frames_dict, 'mean': frames_me}
        """
        frames_dict = data.dict
        frames_in = data.data
        frames_cl = data.label

        # --- Using cell_bib for clustering
        cell_libs_handler = CellLibrary().get_registry()
        libs_class_overview = [lib.split("resort_")[-1] for lib in cell_libs_handler.get_library_overview(do_print=False)]
        libs_use = [f'resort_{lib}' for lib in libs_class_overview if lib in str(self._settings.get_path2folder).lower()]
        if len(libs_use):
            new_data = self.reconfigure_cluster_with_cell_lib(
                fn=cell_libs_handler.build(libs_use[0]),
                sel_mode_classes=self._settings.use_cell_sort_mode,
                dataset=data
            )
            frames_in = new_data.data
            frames_cl = new_data.label
            frames_dict = new_data.dict

        # --- PART: Reducing samples per cluster (if too large)
        if self._settings.reduce_samples_per_cluster_do:
            self._logger.info("... do data augmentation with reducing the samples per cluster")
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
            self._logger.info(f"... do data normalization")
            data_class_frames_in = DataNormalization(self._settings.normalization_method)
            frames_in = data_class_frames_in.normalize(frames_in)

        # --- PART: Mean waveform calculation and data augmentation
        frames_me = calculate_frame_mean(frames_in, frames_cl)

        # --- PART: Data Augmentation
        if self._settings.augmentation_do and not self._settings.reduce_samples_per_cluster_do:
            self._logger.info("... do data augmentation")
            frames_in, frames_cl = augmentation_changing_position(
                frames_in=frames_in,
                frames_cl=frames_cl,
                num_min_frames=self._settings.augmentation_num
            )

        # --- PART: Generate and add noise cluster
        if add_noise_cluster:
            snr_mean = calculate_snr_cluster(frames_in, frames_cl, frames_me)
            snr_range_zero = [np.median(snr_mean[:, 0]), np.median(snr_mean[:, 2])]
            info = np.unique(frames_cl, return_counts=True)
            num_cluster = np.max(info[0]) + 1
            num_frames = np.max(info[1])
            self._logger.info(f"... adding a zero-noise cluster: cluster = {num_cluster} - number of frames = {num_frames}")

            new_frames, new_clusters, new_mean = generate_zero_frames(frames_in.shape[1], num_frames, snr_range_zero)
            frames_in = np.append(frames_in, new_frames, axis=0)
            frames_cl = np.append(frames_cl, num_cluster + new_clusters, axis=0)
            frames_me = np.vstack([frames_me, new_mean])

        return DatasetFromFile(
            data=frames_in,
            label=frames_cl,
            dict=frames_dict,
            mean=frames_me,
        )
