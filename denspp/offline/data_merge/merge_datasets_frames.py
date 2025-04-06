import numpy as np
import platform
from glob import glob
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from datetime import datetime
from tqdm import tqdm
from denspp.offline.data_call.call_handler import SettingsData


class MergeDatasets:
    def __init__(self, pipeline, settings_data: SettingsData, do_list: bool=False) -> None:
        """Class for handling the merging process for generating datasets from transient input signals
        :param pipeline:        Selected pipeline for processing data
        :param settings_data:   Dataclass for handling the transient data
        :param do_list:         Boolean for listing the data
        """
        self.__settings = settings_data
        self.__path2save = settings_data.path

        self.__pipeline = pipeline(self.__settings.fs_resample)
        self.__saving_data_list = do_list
        self.__data_list = list()
        self.__data_single = dict()
        self.__data_merged = dict()
        self.__cluster_available = False

    def __generate_folder(self, addon: str='Merging') -> None:
        """Generating the folder temporary saving"""
        self.__name_temp_folder = addon
        self.path2folder = join(self.__path2save, addon)

        if not self.__saving_data_list:
            makedirs(self.__path2save, exist_ok=True)
            if exists(self.path2folder):
                rmtree(self.path2folder)
            makedirs(self.path2folder, exist_ok=True)
        else:
            if exists(self.path2folder):
                rmtree(self.path2folder)

    def __erase_folder(self, do_erase: bool=True) -> None:
        """Erasing the folder temporary saving"""
        if do_erase:
            rmtree(self.path2folder)

    def __iteration_determine_duration(self, time_start: datetime) -> None:
        """"""
        time_dt = datetime.now() - time_start
        iteration = self.__data_single['ite_recovered']
        print(f"... done after {time_dt.seconds + 1e-6 * time_dt.microseconds: .2f} s"
              f"\n... recovered {iteration} samples")
        self.__output_meta(False)

    def __iteration_save_results(self) -> None:
        """"""
        if self.__saving_data_list:
            self.__data_list.append(self.__data_single)
        else:
            file_name = f"{self.__data_single['file_name']}.npy"
            np.save(file_name, self.__data_single)
            print(f'Saving file in: {file_name}')

    def __output_meta(self, take_merged: bool=True) -> None:
        """Generating print output with meta information"""
        data0 = self.__data_single if not take_merged else self.__data_merged

        meta_infos_frames = data0["frames_in"].shape
        meta_infos_id = np.unique(data0["frames_cl"], return_counts=True)
        if take_merged:
            print(f"\n========================================================"
                  f"\nSummary of merging data"
                  f"\n========================================================")
        print(f"... available frames: {meta_infos_frames[0]} samples with each size of {meta_infos_frames[1]}"
              f'\n... available classes: {meta_infos_id[0]} with {meta_infos_id[1]} samples')

    def get_frames_from_dataset(self, data_loader, cluster_class_avai: bool=False, process_points: list=()) -> None:
        """Tool for loading datasets in order to generate one new dataset (Step 1)
        Args:
            data_loader:        Function with DataLoader
            cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
            process_points:     Taking the datapoints of the choicen dataset [Start, End]
        """
        self.__generate_folder()
        self.__cluster_available = cluster_class_avai
        fs_ana = self.__pipeline.fs_ana
        fs_adc = self.__pipeline.fs_adc

        # --- Setting the points
        do_reduced_sample = len(process_points) > 0
        if do_reduced_sample and len(process_points) > 0:
            runPoint = process_points[0]
            if len(process_points) == 2:
                use_end_point = process_points[1] if process_points[1] else 0
            else:
                use_end_point = 0
        else:
            runPoint = 0
            use_end_point = 0
        endPoint = 0

        # --- Calling the data into RAM (Iterating the files)
        print("... processing data")
        settings = dict()
        first_run = True
        while first_run or runPoint < endPoint:
            self.__data_single = dict()
            first_run = True
            ite_recoverd = 0
            frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
            frames_cl = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))
            time_start = datetime.now()

            # --- Getting data
            self.__settings = runPoint
            datahandler = data_loader(self.__settings)
            datahandler.do_call()
            datahandler.do_resample()

            # --- Processing data (Iterating the channels)
            print(f"\nProcessing data sample {runPoint}:\n========================================================")
            for ch in tqdm(datahandler._raw_data.electrode_id, ncols=100, desc="Progress: "):
                spike_xpos = np.floor(datahandler._raw_data.evnt_xpos[ch] * fs_adc / fs_ana).astype("int")
                # --- Processing the analogue input
                self.__pipeline.run_input(datahandler._raw_data.data_raw[ch], spike_xpos)
                length_data_in = self.__pipeline.signals.x_adc.size

                frame_new = self.__pipeline.signals.frames_align
                frame_cl = datahandler._raw_data.evnt_id[ch]

                # --- Post-Processing: Checking if same length
                if frame_new.shape[0] != frame_cl.size:
                    ite_recoverd += 1
                    # Check where errors are available
                    sample_first_delete_pos = np.argwhere((spike_xpos - self.__pipeline.frame_left_windowsize) <= 0).flatten()
                    sample_first_do_delete = sample_first_delete_pos.size > 0
                    sample_last_delete_pos = np.argwhere(
                        ((spike_xpos + self.__pipeline.frame_right_windowsize) >= length_data_in) < 0).flatten()
                    sample_last_do_delete = sample_last_delete_pos.size > 0

                    if sample_first_do_delete and not sample_last_do_delete:
                        frame_cl = np.delete(frame_cl, sample_first_delete_pos)
                    elif not sample_first_do_delete and sample_last_do_delete:
                        frame_cl = np.delete(frame_cl, sample_last_delete_pos)
                    elif sample_first_do_delete and sample_last_do_delete:
                        frame_cl = np.delete(frame_cl, (sample_first_delete_pos, sample_last_delete_pos))

                    # Only suitable for RGC TDB data (unknown error)
                    elif not sample_first_do_delete and not sample_last_do_delete:
                        if np.unique(frame_cl).size == 1:
                            num_min = np.min((frame_cl.size, frame_new.shape[0]))
                            frame_cl = frame_cl[0:num_min - 1]
                            frame_new = frame_new[0:num_min - 1, ]
                        else:
                            continue

                    # --- Second check
                    if frame_cl.size != frame_new.shape[0]:
                        num_min = np.min((frame_cl.size, frame_new.shape[0]))
                        frame_cl = frame_cl[0:num_min - 1]
                        frame_new = frame_new[0:num_min - 1, ]

                # --- Processing (Frames and cluster)
                if first_run:
                    endPoint = process_points[1] if use_end_point != 0 else datahandler._no_files
                    settings = self.__pipeline.prepare_saving()
                    frames_in = frame_new
                    frames_cl = frame_cl
                else:
                    frames_in = np.concatenate((frames_in, frame_new), axis=0)
                    frames_cl = np.concatenate((frames_cl, frame_cl), axis=0)
                first_run = False

                if frames_in.shape[0] != frames_cl.size:
                    print(f'Data merging has an error after channel #{ch}')

                # --- Release memory
                del spike_xpos, frame_new, frame_cl

            # --- Bringing data into format
            create_time = datetime.now().strftime("%Y-%m-%d")
            file_name = join(self.path2folder, f"{create_time}_Dataset-{datahandler._raw_data.data_name}_step{runPoint + 1:03d}")
            self.__data_single.update({"frames_in": frames_in, "frames_cl": frames_cl, "ite_recovered": ite_recoverd})
            self.__data_single.update({"settings": settings, "num_clusters": np.unique(frames_cl).size})
            self.__data_single.update({"file_name": file_name})

            # --- Last steps in each iteration
            self.__iteration_determine_duration(time_start)
            self.__iteration_save_results()

            # --- Release memory
            self.__pipeline.clean_pipeline()
            del datahandler, frames_in, frames_cl
            runPoint += 1

    def merge_data_from_diff_files(self) -> None:
        """Merging data files from specific folder into one file"""
        split_format = '\\' if platform.system() == "Windows" else '/'

        # --- Linking data
        if self.__saving_data_list:
            # --- Processing internal list from storage
            settings = self.__data_list[-1]['settings']
            data_used = self.__data_list
            file_name = data_used[-1]['file_name'].split(split_format)[-1].split('_step')[0]
        else:
            # --- Processing external *.npy files
            folder_content = glob(join(self.__path2save, self.__name_temp_folder, '*.npy'))
            folder_content.sort()

            settings = dict()
            data_used = folder_content
            file_name = data_used[-1].split(split_format)[-1].split('_step')[0]

        # --- Getting data
        self.__data_merged = dict()
        frame_in = np.zeros((0, 0), dtype='int16')
        frame_cl = np.zeros((0, 0), dtype='uint16')

        max_num_clusters = 0
        for idx, file in enumerate(data_used):
            if not self.__saving_data_list:
                data = np.load(file, allow_pickle=True).item()
                settings = data['settings']
            else:
                data = file

            cl_in = data['frames_cl'] + max_num_clusters
            frame_in = data['frames_in'] if idx == 0 else np.append(frame_in, data['frames_in'], axis=0)
            frame_cl = cl_in if idx == 0 else np.append(frame_cl, cl_in, axis=0)
            max_num_clusters = 0 if self.__cluster_available else 1 + np.unique(frame_cl).max()

        # --- Transfer in common structure
        create_time = datetime.now().strftime("%Y-%m-%d")
        self.__data_merged.update({"data": frame_in, "class": frame_cl})
        self.__data_merged.update({"create_time": create_time, "settings": settings})
        self.__data_merged.update({"file_name": file_name})

    def save_merged_data_in_npyfile(self) -> str:
        """Saving the results in *.npy-file"""
        self.__output_meta(True)
        path2file = join(self.__path2save, self.__data_merged["file_name"]) + "_Merged.npy"
        np.save(path2file, self.__data_merged)
        print(f'Saving file in: {path2file}')
        return path2file
