import numpy as np
import numpy.lib.scimath as sm
from pathlib import Path
from logging import getLogger, Logger
from collections import defaultdict
from glob import glob
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from denspp.offline import get_path_to_project, check_keylist_elements_any
from denspp.offline.data_call import SettingsData, DataFromFile
from denspp.offline.preprocessing import FrameWaveform
from denspp.offline.metric.data_numpy import calculate_error_mse


class MergeDataset:
    def __init__(self, pipeline, dataloader, settings_data: SettingsData, concatenate_id: bool=False) -> None:
        """Class for handling the merging process for generating datasets from transient input signals
        :param pipeline:                Construct of selected pipeline for processing data
        :param dataloader:              Construct of used Dataloader for getting and handling data
        :param settings_data:           Class SettingsData for configuring the data loader
        :param concatenate_id:          Do concatenation of the class number with increasing id number (useful for non-biological clusters)
        :return:                        None
        """
        self._logger: Logger = getLogger(__name__)

        self._dataloader = dataloader
        self._settings: SettingsData = settings_data
        self._pipeline = pipeline
        self._check_right_pipeline()

        self._path2save = get_path_to_project('temp_merge')
        self._do_label_concatenation = concatenate_id

    def _check_right_pipeline(self) -> None:
        if not check_keylist_elements_any(
            keylist=dir(self._pipeline),
            elements=['run_preprocessing', 'run_classifier', 'settings']
        ):
            raise ImportError("Wrong pipeline is implemented. It should include 'run_preprocessing' and 'run_classifier'.")

    def _generate_folder(self) -> None:
        if exists(self._path2save):
            rmtree(self._path2save)
        makedirs(self._path2save, exist_ok=True)

    def _save_results(self, data: dict, path2folder: str, file_name: str) -> str:
        file_name = f"{file_name}.npy"
        path2file = join(path2folder, file_name)
        np.save(path2file, data, allow_pickle=True)
        self._logger.info(f'Saving file in: {file_name}')
        return path2file

    def _iteration_save_results(self, data: list, data_name: str) -> None:
        create_time = datetime.now().strftime("%Y-%m-%d")
        data_save = {
            "frames": data
        }
        self._save_results(
            data=data_save,
            path2folder=self._path2save,
            file_name=f"{create_time}_Dataset-{data_name}"
        )

    def _get_frames_from_labeled_dataset(self, data: DataFromFile, xpos_offset: int=0) -> list:
        self._logger.info(f"\nProcessing file: {data.data_name}")
        pipeline = self._pipeline(data.fs_used, False)

        frames_extracted = list()
        for rawdata, xposition, label in tqdm(zip(data.data_raw, data.evnt_xpos, data.evnt_id), ncols=100, desc="Progress: "):
            xpos_scaler = pipeline.fs_ana / pipeline.fs_ana
            xpos_updated = np.floor(xpos_scaler * xposition).astype("int")
            result = pipeline.run_preprocessor(rawdata, xpos_updated, xpos_offset)

            frame_new: FrameWaveform = result['frames']
            frame_new.label = label
            frames_extracted.append(frame_new)
            del frame_new
        return frames_extracted

    def _get_frames_from_unlabeled_dataset(self, data: DataFromFile, **kwargs) -> list:
        self._logger.info(f"\nProcessing file: {data.data_name}")
        pipeline = self._pipeline(data.fs_used, False)

        frames_extracted = list()
        for rawdata in tqdm(data.data_raw, ncols=100, desc="Progress: "):
            result = pipeline.run_preprocessor(rawdata)

            frame_new: FrameWaveform = result['frames']
            frames_extracted.append(frame_new)
            del frame_new
        return frames_extracted

    def get_frames_from_dataset(self, process_points: list=(), xpos_offset: int=0) -> None:
        """Tool for loading datasets in order to generate one new dataset (Step 1)
        :param process_points:      Taking the datapoints of the selected data set to process
        :param xpos_offset:         Integer as position offset for shifting label position of an event (only apply if label exists)
        :return:                    None
        """
        self._generate_folder()
        current_index = 0
        while True:
            try:
                sets0 = deepcopy(self._settings)
                sets0.data_point = current_index if not len(process_points) else process_points[current_index]
                datahandler = self._dataloader(sets0)
                datahandler.do_call()
            except:
                break
            else:
                datahandler.do_resample()
                datahandler.do_cut()
                data: DataFromFile = datahandler.get_data()
                del datahandler

                if data.label_exist:
                    result = self._get_frames_from_labeled_dataset(data, xpos_offset=xpos_offset)
                else:
                    result = self._get_frames_from_unlabeled_dataset(data)
                self._iteration_save_results(
                    data=result,
                    data_name=data.data_name
                )
                current_index += 1

    def merge_data_from_all_iteration(self) -> str:
        """Merging extracted information from all runs into one file
        :return:    String with path to final file
        """
        folder_content = glob(join(self._path2save, '*.npy'))
        folder_content.sort()

        file_name = folder_content[-1]
        dataset_loaded = list()
        for file in folder_content:
            val = np.load(file, allow_pickle=True).item()['frames']
            dataset_loaded.append(val)

        # --- Merging data from different sources
        frames_waveform = list()
        frames_position = list()
        frames_label = list()
        max_num_clusters = 0
        for data_case in dataset_loaded:
            for data_elec in data_case:
                frames_waveform.extend(data_elec.waveform.tolist())
                frames_position.extend(data_elec.xpos)
                frames_label.extend(data_elec.label + max_num_clusters)
                max_num_clusters += 0 if self._do_label_concatenation or len(frames_label) == 0 else 1 + max(frames_label)

        # --- Save output
        data_merged = {
            "data": np.array(frames_waveform),
            "class": np.array(frames_label),
            "position": np.array(frames_position),
            "create_time": datetime.now().strftime("%Y-%m-%d")
        }
        self._save_results(
            data=data_merged,
            path2folder=get_path_to_project('dataset'),
            file_name=Path(file_name).stem + "_Merged"
        )
        return self._path2save
