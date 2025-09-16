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
from denspp.offline.data_call import SettingsData, DataHandler
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

    def _get_frames_from_labeled_dataset(self, data: DataHandler, xpos_offset: int=0) -> list:
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

    def _get_frames_from_unlabeled_dataset(self, data: DataHandler, **kwargs) -> list:
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
                data: DataHandler = datahandler.get_data()
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


def _crossval(wave1, wave2):
    result = np.correlate(wave1, wave2, 'full') / (sm.sqrt(sum(np.square(wave1))) * sm.sqrt(sum(np.square(wave2))))
    return result


def _calc_metric(wave_in, wave_ref):
    maxIn = max(wave_in)
    maxInIndex = wave_in.argmax()
    maxRef = max(wave_ref)
    maxRefIndex = wave_ref.argmax()

    result = list()
    result.append(maxInIndex - maxRefIndex)
    result.append(calculate_error_mse(wave_in, wave_ref))
    result.append(np.abs(np.trapezoid(wave_in[:maxInIndex + 1]) - np.trapezoid(wave_in[maxInIndex:])))
    result.append(maxInIndex)
    result.append(maxIn)
    return np.array(result)


class SortDataset:
    def __init__(self, path_2_file: str) -> None:
        """Tool for loading and processing dataset to generate a sorted dataset"""
        self._logger: Logger = getLogger(__name__)
        self.setOptions = dict()
        self.setOptions['do_2nd_run'] = False
        self.setOptions['do_resort'] = True
        self.addon = '_Sorted'
        self.setOptions['path2file'] = path_2_file
        self.setOptions['path2save'] = path_2_file[:len(path_2_file) - 4] + self.addon + '.mat'
        self.setOptions['path2fig'] = path_2_file[:len(path_2_file) - 4]

        if "Martinez" in path_2_file:
            self.criterion_CheckDismiss = [3, 0.7]
            self.criterion_Run0 = 0.98
            self.criterion_Resort = 0.98
        elif "Quiroga" in path_2_file:
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
        self._logger.info("Start of sorting the dataset")

        if frames_cluster.shape[0] == 1:
            frames_cluster = np.transpose(frames_cluster)

        frames_cluster = frames_cluster.tolist()
        frames_cluster = [item for sublist in frames_cluster for item in sublist]

        data_raw_pos, data_raw_frames, data_raw_means, data_raw_metric, input_cluster, data_raw_number = (
            self.prepare_data(frames_cluster, frames_in))

        self._logger.info('... data loaded and pre-selected')
        del frames_in, frames_cluster, mat_file
        # endregion

        # region Pre-Processing: Consistency Check
        self._logger.info('Consistency check of the frames')
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
        self._logger.info(" ... end of step")
        # endregion

        # region Processing: Merging Cluster
        self._logger.info("Merging clusters")
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

            # Erste Prüfung: Mean-Waveform vergleichen mit bereits vorliegenden Clustern
            metric_Run0 = [_calc_metric(_crossval(Ymean_New, Ycheck_Mean), _crossval(Ycheck_Mean, Ycheck_Mean)) for
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
                metric_Run1 = np.array([_calc_metric(_crossval(Y, YMean), WaveRef) for Y in YCheck])
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
        self._logger.info(" ... end of step")

        data_dismiss_XCheck[len(data_dismiss_XCheck)] = data_missed_new_XCheck.get(len(data_dismiss_XCheck) + 1,
                                                                                   np.array([]))
        data_dismiss_YCheck[len(data_dismiss_YCheck)] = data_missed_new_YCheck.get(len(data_dismiss_YCheck) + 1,
                                                                                   np.array([]))
        data_dismiss_mean[len(data_dismiss_mean)] = np.mean(
            data_missed_new_YCheck.get(len(data_dismiss_mean) + 1, np.array([])), axis=0, dtype=np.float64)

        for idy, Yraw in data_2merge_YCheck.items():
            WaveRef = _crossval(data_2merge_mean[idy], data_2merge_mean[idy])
            data_2merge_metric[idy] = [_calc_metric(_crossval(Y, data_2merge_mean[idy]), WaveRef) for Y in Yraw]

        del idx, idy, candX, candY, Yraw_New, Ymean_New, Xraw_New, WaveRef, Yraw, \
            data_missed_new_YCheck, data_missed_new_XCheck
        # endregion

        # region Post-Processing: Resorting dismissed frames
        data_restored = 0
        if self.setOptions['do_resort']:
            self._logger.info("Resorting dismissed frames")
            for idz, value in tqdm(enumerate(data_dismiss_XCheck), ncols=100, desc="Resorted Frames: ", unit="frames"):
                pos_sel = data_dismiss_XCheck[value]
                frames_sel = data_dismiss_YCheck[value]

                for idx in range(frames_sel.shape[0]):
                    metric_Run2 = np.array([_calc_metric(_crossval(frames_sel[idx, :], data_2merge_mean[idx2]),
                                                         _crossval(data_2merge_mean[idx2], data_2merge_mean[idx2]))
                                            for idx2 in range(data_2merge_number)])
                    # Decision
                    selY, selX = np.max(metric_Run2[:, 4]), np.argmax(metric_Run2[:, 4])
                    if selY >= self.criterion_Resort:
                        data_2merge_XCheck[selX] = np.vstack([data_2merge_XCheck[selX], pos_sel[idx, :]])
                        data_2merge_YCheck[selX] = np.vstack([data_2merge_YCheck[selX], frames_sel[idx, :]])
                        data_restored += 1
            self._logger.info(" ... end of step")
        del idz, value, pos_sel, frames_sel, idx, metric_Run2, selY, selX
        # endregion

        # region Preparing: Transfer to new file
        output, data_process_num = self.prepare_data_for_saving(data_2merge_XCheck, data_2merge_YCheck)
        self.save_output_as_npyfile(output, data_process_num, frames_in_number)
        self._logger.info(" ... merged output generated")

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
                _calc_metric(_crossval(YCheck[idy, :], np.mean(YCheck, axis=0, dtype=np.float64)), WaveRef)
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
                calc_temp = np.append(_calc_metric(WaveIn, WaveRef), XCheck[idy])
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
            metric_Check1.append(_calc_metric(WaveIn, WaveRef))
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
        self._logger.info(f"Percentage of overall kept frames: {data_ratio_merged * 100:.2f}")
        self._logger.info(f"Percentage of overall dismissed frames: {data_ratio_dismiss * 100:.2f}")
        np.save(self.setOptions['path2save'], out, allow_pickle=True)
