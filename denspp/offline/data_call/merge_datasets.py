import numpy as np
import numpy.lib.scimath as sm
import platform
from logging import getLogger, Logger
from collections import defaultdict
from glob import glob
from os import makedirs
from os.path import join, exists
from shutil import rmtree
from datetime import datetime
from tqdm import tqdm
from copy import deepcopy

from denspp.offline import get_path_to_project
from denspp.offline.data_call import SettingsData
from denspp.offline.metric.data_numpy import calculate_error_mse
from denspp.offline.metric.snr import calculate_snr


class MergeDataset:
    _cluster_available: bool=False
    _data_list = list()
    _data_single = dict()
    _data_merged = dict()

    def __init__(self, pipeline, dataloader, settings_data: SettingsData, do_list: bool=False) -> None:
        """Class for handling the merging process for generating datasets from transient input signals
        :param pipeline:            Selected pipeline for processing data
        :param dataloader:   Used Dataloader for getting and handling data
        :param settings_data:       Dataclass for handling the transient data
        :param do_list:             Boolean for listing the data
        """
        self._logger: Logger = getLogger(__name__)
        self._settings = settings_data
        self._path2save = get_path_to_project()
        self._pipeline = pipeline
        self._dataloader = dataloader
        self._saving_data_list = do_list

    def __generate_folder(self, addon: str='Merging') -> None:
        """Generating the folder temporary saving"""
        self.__name_temp_folder = addon
        self.path2folder = join(self._path2save, addon)

        if not self._saving_data_list:
            makedirs(self._path2save, exist_ok=True)
            if exists(self.path2folder):
                rmtree(self.path2folder)
            makedirs(self.path2folder, exist_ok=True)
        else:
            if exists(self.path2folder):
                rmtree(self.path2folder)

    def __iteration_determine_duration(self, time_start: datetime) -> None:
        """"""
        time_dt = datetime.now() - time_start
        iteration = self._data_single['ite_recovered']
        print(f"... done after {time_dt.seconds + 1e-6 * time_dt.microseconds: .2f} s"
              f"\n... recovered {iteration} samples")
        self.__output_meta(False)

    def __iteration_save_results(self) -> None:
        """"""
        if self._saving_data_list:
            self._data_list.append(self._data_single)
        else:
            file_name = f"{self._data_single['file_name']}.npy"
            np.save(file_name, self._data_single)
            print(f'Saving file in: {file_name}')

    def __output_meta(self, take_merged: bool=True) -> None:
        """Generating print output with meta information"""
        data0 = self._data_single if not take_merged else self._data_merged

        meta_infos_frames = data0["frames_in"].shape
        meta_infos_id = np.unique(data0["frames_cl"], return_counts=True)
        if take_merged:
            print(f"\n========================================================"
                  f"\nSummary of merging data"
                  f"\n========================================================")
        print(f"... available frames: {meta_infos_frames[0]} samples with each size of {meta_infos_frames[1]}"
              f'\n... available classes: {meta_infos_id[0]} with {meta_infos_id[1]} samples')

    def get_frames_from_dataset(self, concatenate_id: bool=False, process_points: list=()) -> None:
        """Tool for loading datasets in order to generate one new dataset (Step 1)
        :param concatenate_id:      Do concatenation of the class number with increasing id number (useful for non-biological clusters)
        :param process_points:      Taking the datapoints of the choicen dataset [Start, End]
        """
        self.__generate_folder()
        self._cluster_available = concatenate_id
        fs_ana = self._pipeline.fs_ana
        fs_adc = self._pipeline.fs_adc

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
        used_pipe = self._pipeline()
        settings = dict()
        first_run = True
        while first_run or runPoint < endPoint:
            self._data_single = dict()
            first_run = True
            ite_recoverd = 0
            frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
            frames_cl = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))
            time_start = datetime.now()

            # --- Getting data
            sets0 = deepcopy(self._settings)
            sets0.data_case = runPoint
            datahandler = self._dataloader(sets0)
            datahandler.do_call()
            datahandler.do_resample()
            data = datahandler.get_data()
            del datahandler

            # --- Processing data (Iterating the channels)
            print(f"\nProcessing data sample {runPoint}:\n========================================================")
            for ch, id in tqdm(enumerate(data.electrode_id), ncols=100, desc="Progress: "):
                spike_xpos = np.floor(data.evnt_xpos[ch] * fs_adc / fs_ana).astype("int")
                # --- Processing the analogue input
                data_rslt = self._pipeline.run_input(data.data_raw[ch, :], spike_xpos)
                length_data_in = data_rslt['x_adc'].size

                frame_new = data_rslt['frames_align']
                frame_cl = data.evnt_id[ch]

                # --- Post-Processing: Checking if same length
                if frame_new.shape[0] != frame_cl.size:
                    ite_recoverd += 1
                    # Check where errors are available
                    sample_first_delete_pos = np.argwhere((spike_xpos - self._pipeline.frame_left_windowsize) <= 0).flatten()
                    sample_first_do_delete = sample_first_delete_pos.size > 0
                    sample_last_delete_pos = np.argwhere(
                        ((spike_xpos + self._pipeline.frame_right_windowsize) >= length_data_in) < 0).flatten()
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
                    settings = self._pipeline.prepare_saving()
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
            file_name = join(self.path2folder, f"{create_time}_Dataset-{data.data_name}_step{runPoint + 1:03d}")
            self._data_single.update({"frames_in": frames_in, "frames_cl": frames_cl, "ite_recovered": ite_recoverd})
            self._data_single.update({"settings": settings, "num_clusters": np.unique(frames_cl).size})
            self._data_single.update({"file_name": file_name})

            # --- Last steps in each iteration
            self.__iteration_determine_duration(time_start)
            self.__iteration_save_results()

            # --- Release memory
            del datahandler, frames_in, frames_cl
            runPoint += 1

    def merge_data_from_diff_files(self) -> None:
        """Merging data files from specific folder into one file"""
        split_format = '\\' if platform.system() == "Windows" else '/'

        # --- Linking data
        if self._saving_data_list:
            # --- Processing internal list from storage
            settings = self._data_list[-1]['settings']
            data_used = self._data_list
            file_name = data_used[-1]['file_name'].split(split_format)[-1].split('_step')[0]
        else:
            # --- Processing external *.npy files
            folder_content = glob(join(self._path2save, self.__name_temp_folder, '*.npy'))
            folder_content.sort()

            settings = dict()
            data_used = folder_content
            file_name = data_used[-1].split(split_format)[-1].split('_step')[0]

        # --- Getting data
        self._data_merged = dict()
        frame_in = np.zeros((0, 0), dtype='int16')
        frame_cl = np.zeros((0, 0), dtype='uint16')

        max_num_clusters = 0
        for idx, file in enumerate(data_used):
            if not self._saving_data_list:
                data = np.load(file, allow_pickle=True).item()
                settings = data['settings']
            else:
                data = file

            cl_in = data['frames_cl'] + max_num_clusters
            frame_in = data['frames_in'] if idx == 0 else np.append(frame_in, data['frames_in'], axis=0)
            frame_cl = cl_in if idx == 0 else np.append(frame_cl, cl_in, axis=0)
            max_num_clusters = 0 if self._cluster_available else 1 + np.unique(frame_cl).max()

        # --- Transfer in common structure
        create_time = datetime.now().strftime("%Y-%m-%d")
        self._data_merged.update({"data": frame_in, "class": frame_cl})
        self._data_merged.update({"create_time": create_time, "settings": settings})
        self._data_merged.update({"file_name": file_name})

    def save_merged_data_in_npyfile(self) -> str:
        """Saving the results in *.npy-file"""
        self.__output_meta(True)
        path2file = join(self._path2save, self._data_merged["file_name"]) + "_Merged.npy"
        np.save(path2file, self._data_merged)
        print(f'Saving file in: {path2file}')
        return path2file


def _crossval(wave1, wave2):
    result = np.correlate(wave1, wave2, 'full') / (sm.sqrt(sum(np.square(wave1))) * sm.sqrt(sum(np.square(wave2))))
    return result


def calc_metric(wave_in, wave_ref):
    maxIn = max(wave_in)
    maxInIndex = wave_in.argmax()
    maxRef = max(wave_ref)
    maxRefIndex = wave_ref.argmax()

    result = []
    result.append(maxInIndex - maxRefIndex)
    result.append(calculate_error_mse(wave_in, wave_ref))
    result.append(np.abs(np.trapz(wave_in[:maxInIndex + 1]) - np.trapz(wave_in[maxInIndex:])))
    result.append(maxInIndex)
    result.append(maxIn)
    return np.array(result)


class SortDataset:
    def __init__(self, path_2_file: str):
        """Tool for loading and processing dataset to generate a sorted dataset"""
        self.setOptions = dict()
        self.setOptions['do_2nd_run'] = False
        self.setOptions['do_resort'] = True
        self.addon = '_Sorted'
        self.setOptions['path2file'] = path_2_file
        self.setOptions['path2save'] = path_2_file[:len(path_2_file) - 4] + self.addon + '.mat'
        self.setOptions['path2fig'] = path_2_file[:len(path_2_file) - 4]

        if "Martinez" in path_2_file:
            # Settings Martinez
            self.criterion_CheckDismiss = [3, 0.7]
            self.criterion_Run0 = 0.98
            self.criterion_Resort = 0.98
        elif "Quiroga" in path_2_file:
            # Settings für Quiroga
            self.criterion_CheckDismiss = [2, 0.96]
            self.criterion_Run0 = 0.98
            self.criterion_Resort = 0.95
        else:
            self.criterion_CheckDismiss = [3, 0.7]
            self.criterion_Run0 = 0.98
            self.criterion_Resort = 0.98

    def _loading_data(self):
        return np.load(self.setOptions['path2file'], allow_pickle=True)

    def sort_dataset(self):
        """Sort the frames to each cluster and dismiss unfitting frames"""
        # region Pre-Processing: Input structuring
        mat_file = self._loading_data()

        frames_cluster = mat_file['frames_cl']
        frames_in = mat_file['frames_in']
        frames_in_number = frames_in.shape[0]
        print("Start of sorting the dataset")

        if frames_cluster.shape[0] == 1:
            frames_cluster = np.transpose(frames_cluster)

        frames_cluster = frames_cluster.tolist()
        frames_cluster = [item for sublist in frames_cluster for item in sublist]

        data_raw_pos, data_raw_frames, data_raw_means, data_raw_metric, input_cluster, data_raw_number = (
            self.prepare_data(frames_cluster, frames_in))

        print('\n... data loaded and pre-selected')
        del frames_in, frames_cluster, mat_file
        # endregion

        # region Pre-Processing: Consistency Check
        print('Consistency check of the frames')
        data_process_XCheck = defaultdict()  # data_1process[2]
        data_process_YCheck = defaultdict()  # data_1process[3]
        data_process_mean = defaultdict()  # data_1process[4]
        data_process_metric = defaultdict()  # data_1process[5]
        data_dismiss_XCheck = defaultdict()
        data_dismiss_YCheck = defaultdict()
        data_dismiss_mean = defaultdict()
        data_dismiss_metric = defaultdict()

        for idx in tqdm(input_cluster, ncols=100, desc="Checked Cluster: ", unit="cluster"):
            YCheckIn, XCheckIn, XCheck, XCheck_False = self.split_frames(idx, data_raw_frames, data_raw_pos)
            # Übergabe: processing frames
            data_process_XCheck[idx], data_process_YCheck[idx], data_process_mean[idx], data_process_metric[idx] = (
                self.get_frames(idx, YCheckIn, XCheckIn, XCheck))

            # Übergabe: dismissed frames
            data_dismiss_XCheck[idx], data_dismiss_YCheck[idx], data_dismiss_mean[idx], data_dismiss_metric[idx] = (
                self.get_frames(idx, XCheckIn, XCheckIn, XCheck_False))

        del idx, YCheckIn, XCheckIn, XCheck, XCheck_False
        print(" ... end of step")
        # endregion

        # region Processing: Merging Cluster
        print("Merging clusters")
        data_2merge_XCheck = defaultdict()
        data_2merge_YCheck = defaultdict()
        data_2merge_mean = defaultdict()
        data_2merge_metric = defaultdict()
        data_2wrong_Xnew = defaultdict()
        data_2wrong_Ynew = defaultdict()
        data_2wrong_mean = defaultdict()

        data_2merge_XCheck[0] = data_process_XCheck[0]  # data_2merge[2]
        data_2merge_YCheck[0] = data_process_YCheck[0]  # data_2merge_[3]
        data_2merge_mean[0] = data_process_mean[0]  # data_2merge[4]

        data_2merge_number = 0
        data_2wrong_number = 0
        data_missed_new_XCheck = defaultdict()
        data_missed_new_YCheck = defaultdict()

        for idx in tqdm(range(1, len(input_cluster)), ncols=100, desc="Sorted Cluster: ", unit="cluster"):
            Yraw_New = data_process_YCheck[idx]
            Ymean_New = data_process_mean[idx]
            Xraw_New = data_process_XCheck[idx]

            # Erste Prüfung: Mean-Waveform vergleichen mit bereits gemergten Clustern
            metric_Run0 = [calc_metric(_crossval(Ymean_New, Ycheck_Mean), _crossval(Ycheck_Mean, Ycheck_Mean)) for
                           Ycheck_Mean in data_2merge_mean.values()]

            # Entscheidung treffen
            candY = np.max(np.array(metric_Run0)[:, 4])
            candX = np.argmax(np.array(metric_Run0)[:, 4])
            if np.isnan(candX):
                # Keine Lösung vorhanden: Anhängen
                data_2wrong_number += 1
                data_2wrong_Xnew[idx] = Xraw_New
                data_2wrong_Ynew[idx] = Yraw_New
                data_2wrong_mean[idx] = Ymean_New
            elif metric_Run0[candX][4] >= self.criterion_Run0:
                # Zweite Prüfung: Einzel-Waveform mit Mean
                YCheck = np.vstack([data_2merge_YCheck[candX], Yraw_New])
                XCheck = np.vstack([data_2merge_XCheck[candX], Xraw_New])
                YMean = data_2merge_mean[candX]
                WaveRef = _crossval(YMean, YMean)
                metric_Run1 = np.array([calc_metric(_crossval(Y, YMean), WaveRef) for Y in YCheck])
                selOut = np.where(metric_Run1[:, 4] <= 0.92)[0]
                if selOut.size != 0:
                    data_missed_new_XCheck[len(data_dismiss_XCheck) + 1] = np.vstack(
                        [data_missed_new_XCheck[idx], XCheck[selOut, :]])
                    data_missed_new_YCheck[len(data_dismiss_YCheck) + 1] = np.vstack(
                        [data_missed_new_YCheck[idx], YCheck[selOut, :]])

                    XCheck = np.delete(XCheck, selOut, axis=0)
                    YCheck = np.delete(YCheck, selOut, axis=0)

                # Potentieller Match
                data_2merge_XCheck[candX] = XCheck
                data_2merge_YCheck[candX] = YCheck
                data_2merge_mean[candX] = np.mean(YCheck, axis=0, dtype=np.float64)
            else:
                # Neues Cluster
                data_2merge_number += 1
                data_2merge_XCheck[data_2merge_number] = Xraw_New
                data_2merge_YCheck[data_2merge_number] = Yraw_New
                data_2merge_mean[data_2merge_number] = Ymean_New
        print(" ... end of step")

        data_dismiss_XCheck[len(data_dismiss_XCheck)] = data_missed_new_XCheck.get(len(data_dismiss_XCheck) + 1,
                                                                                   np.array([]))
        data_dismiss_YCheck[len(data_dismiss_YCheck)] = data_missed_new_YCheck.get(len(data_dismiss_YCheck) + 1,
                                                                                   np.array([]))
        data_dismiss_mean[len(data_dismiss_mean)] = np.mean(
            data_missed_new_YCheck.get(len(data_dismiss_mean) + 1, np.array([])), axis=0, dtype=np.float64)

        for idy, Yraw in data_2merge_YCheck.items():
            WaveRef = _crossval(data_2merge_mean[idy], data_2merge_mean[idy])
            data_2merge_metric[idy] = [calc_metric(_crossval(Y, data_2merge_mean[idy]), WaveRef) for Y in Yraw]

        del idx, idy, candX, candY, Yraw_New, Ymean_New, Xraw_New, WaveRef, Yraw, \
            data_missed_new_YCheck, data_missed_new_XCheck
        # endregion

        # region Post-Processing: Resorting dismissed frames
        data_restored = 0
        if self.setOptions['do_resort']:
            print("Resorting dismissed frames")
            for idz, value in tqdm(enumerate(data_dismiss_XCheck), ncols=100, desc="Resorted Frames: ", unit="frames"):
                pos_sel = data_dismiss_XCheck[value]
                frames_sel = data_dismiss_YCheck[value]

                for idx in range(frames_sel.shape[0]):
                    metric_Run2 = np.array([calc_metric(_crossval(frames_sel[idx, :], data_2merge_mean[idx2]),
                                                        _crossval(data_2merge_mean[idx2], data_2merge_mean[idx2]))
                                            for idx2 in range(data_2merge_number)])
                    # Decision
                    selY, selX = np.max(metric_Run2[:, 4]), np.argmax(metric_Run2[:, 4])
                    if selY >= self.criterion_Resort:
                        data_2merge_XCheck[selX] = np.vstack([data_2merge_XCheck[selX], pos_sel[idx, :]])
                        data_2merge_YCheck[selX] = np.vstack([data_2merge_YCheck[selX], frames_sel[idx, :]])
                        data_restored += 1
            print(" ... end of step")
        del idz, value, pos_sel, frames_sel, idx, metric_Run2, selY, selX
        # endregion

        # region Preparing: Transfer to new file
        output, data_process_num = self.prepare_data_for_saving(data_2merge_XCheck, data_2merge_YCheck)
        self.save_output_as_npyfile(output, data_process_num, frames_in_number)
        print(" ... merged output generated")

    def prepare_data(self, cluster: list, frames: list):
        """Prepare the frames and clusters from matlab file for further processing"""
        data_pos = defaultdict()  # data_raw{2}
        data_frames = defaultdict()  # data_raw{3}
        data_means = defaultdict()  # data_raw{4}
        data_metric = defaultdict()  # data_raw{5}
        unique_cluster = list(set(cluster))  # keeps order
        data_number = 0

        for value in tqdm(unique_cluster, desc="Prepared clusters: ", unit="cluster"):
            pos_in = np.array([i for i, val in enumerate(cluster) if val == value])
            data_pos[value] = pos_in
            data_frames[value] = frames[pos_in, :]
            data_means[value] = np.mean(frames[pos_in, :], axis=0, dtype=np.float64)
            YCheck = data_frames[value]
            WaveRef = _crossval(np.mean(YCheck, axis=0, dtype=np.float64), np.mean(YCheck, axis=0, dtype=np.float64))
            metric_Check = [
                calc_metric(_crossval(YCheck[idy, :], np.mean(YCheck, axis=0, dtype=np.float64)), WaveRef)
                for
                idy in range(len(pos_in))]
            data_metric[value] = metric_Check
            data_number += len(pos_in)

        return data_pos, data_frames, data_means, data_metric, unique_cluster, data_number

    def split_frames(self, idx: int, data_raw_frames, data_raw_pos):
        """Splitting frames into frames to be processed and to be dismissed"""
        do_run = True
        YCheckIn = data_raw_frames[idx]
        XCheckIn = data_raw_pos[idx]
        XCheck = np.arange(0, len(XCheckIn))
        XCheck_False = []
        IteNo = 0
        while do_run and len(XCheck) > 0:
            YCheck = YCheckIn[XCheck, :]
            metric_Check = list()
            mean_wfg = np.mean(YCheck, axis=0, dtype=np.float64)
            WaveRef = _crossval(mean_wfg, mean_wfg)
            for idy, Y in enumerate(YCheck):
                WaveIn = _crossval(Y, mean_wfg)
                calc_temp = np.append(calc_metric(WaveIn, WaveRef), XCheck[idy])
                metric_Check.append(calc_temp)

            metric_Check = np.array(metric_Check)
            criteria = (np.abs(metric_Check[:, 0]) > self.criterion_CheckDismiss[0]) | (
                    metric_Check[:, 4] < self.criterion_CheckDismiss[1])
            check = np.where(criteria)[0]
            XCheck_False.extend(XCheck[check])
            if len(check) > 0 and IteNo <= 100:
                do_run = True
                IteNo = IteNo + 1
            else:
                do_run = False
                IteNo = IteNo
            XCheck = np.delete(XCheck, check)

        tqdm.write(f"\n{len(XCheck_False)} out of {XCheckIn.shape[0]} frames from cluster {idx} will be dismissed")
        return YCheckIn, XCheckIn, XCheck, XCheck_False

    def get_frames(self, idx, YCheckIn, XCheckIn, XCheck):
        """Get the frames at specific positions"""
        metric_Check1 = list()
        YCheck = YCheckIn[XCheck, :]
        mean_wfg = np.mean(YCheck, axis=0)
        WaveRef = _crossval(mean_wfg, mean_wfg)
        for idy, value in enumerate(XCheck):
            WaveIn = _crossval(YCheck[idy, :], mean_wfg)
            metric_Check1.append(calc_metric(WaveIn, WaveRef))
        return (np.column_stack((XCheckIn[XCheck], idx + np.ones(len(XCheckIn[XCheck])))), YCheckIn[XCheck, :],
                mean_wfg, metric_Check1)

    def prepare_data_for_saving(self, data_x: defaultdict, data_y: defaultdict):
        """Prepare the processed frames for saving as a matlab file"""
        create_time = datetime.now().strftime("%Y-%m-%d")
        output = {'data': np.empty((0, 32), dtype=np.float64), 'class': np.empty((0,), dtype=np.int16),
                  'dict': dict(), "create_time": create_time, "settings": self.setOptions}
        data_process_num = 0
        tqdm.write("preparing data for saving")

        for idx, value in tqdm(enumerate(data_x), ncols=100, desc="Saved Frames: ", unit="frames"):
            X = np.array(value, dtype=np.int16) * np.ones(len(data_x[value]), dtype=np.int16)
            Z = data_y[value]
            output['data'] = np.concatenate((output['data'], Z))
            output['class'] = np.concatenate((output['class'], X))

            data_process_num += len(X)
        return output, data_process_num

    def save_output_as_npyfile(self, out: dict, processed_num: int, frames_in_num: int) -> None:
        """Saving output as a numpy file"""
        data_ratio_merged = processed_num / frames_in_num
        data_ratio_dismiss = 1 - data_ratio_merged
        out['data_ratio_merged'] = data_ratio_merged
        print(f"Percentage of overall kept frames: {data_ratio_merged * 100:.2f}")
        print(f"Percentage of overall dismissed frames: {data_ratio_dismiss * 100:.2f}")
        np.save(self.setOptions['path2save'], out, allow_pickle=True)
