from glob import glob
from os import mkdir
from os.path import join, exists

import numpy as np
from datetime import datetime
from scipy.io import savemat, loadmat
from tqdm import tqdm
import platform

from package.data_call.call_spike_files import DataLoader, SettingsDATA
from src_neuro.pipeline_data import Settings, Pipeline


class MergeDatasets:
    def __init__(self, settings_data: SettingsDATA, path2save: str):
        self.__settings = settings_data
        self.path2save = path2save
        self.filepath = path2save
        self.__generate_folder()

    def get_filepath(self) -> str:
        return self.filepath

    def __generate_folder(self, addon='Merging') -> None:
        if not exists(self.path2save):
            mkdir(self.path2save)

        path2folder = join(self.path2save, addon)
        if not exists(path2folder):
            mkdir(path2folder)
        self.path2folder = path2folder

    def get_frames_from_dataset(self, cluster_class_avai=False, process_points=None) -> None:
        """
        Tool for loading datasets in order to generate one new dataset (Step 1),
        cluster_class_avai: False = Concatenate the class number with increasing id number (useful for non-biological clusters)
        only_pos: Taking the datapoints of the choicen dataset [Start, End]
        """
        # --- Loading the src_neuro
        afe_set = Settings(self.__settings.fs_resample)
        fs_ana = afe_set.SettingsADC.fs_ana
        fs_adc = afe_set.SettingsADC.fs_adc

        # --- Setting the points
        do_reduced_sample = isinstance(process_points, list)
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

        # --- Calling the data into RAM
        print("... loading the datasets")
        settings = dict()
        first_run = True
        while first_run or runPoint < endPoint:
            first_run = True
            ite_recoverd = 0
            time_start = datetime.now()

            frames_in = np.empty(shape=(0, 0), dtype=np.dtype('int16'))
            frames_cl = np.empty(shape=(0, 0), dtype=np.dtype('uint16'))

            self.__settings.data_point = runPoint
            datahandler = DataLoader(self.__settings)
            datahandler.do_call()
            datahandler.do_resample()

            # --- Taking signals from handler
            for ch in tqdm(datahandler.raw_data.electrode_id, ncols=100, desc="Progress: "):
                spike_xpos = np.floor(datahandler.raw_data.evnt_xpos[ch] * fs_adc / fs_ana).astype("int")
                # --- Processing the analogue input
                afe = Pipeline(fs_ana)
                afe.run_input(datahandler.raw_data.data_raw[ch], spike_xpos)
                length_data_in = afe.signals.x_adc.size

                frame_new = afe.signals.frames_align
                frame_cl = datahandler.raw_data.evnt_cluster_id[ch]

                # --- Post-Processing: Checking if same length
                if frame_new.shape[0] != frame_cl.size:
                    ite_recoverd += 1
                    # Check where errors are available
                    sample_first_delete_pos = np.argwhere((spike_xpos - afe.frame_left_windowsize) <= 0).flatten()
                    sample_first_do_delete = sample_first_delete_pos.size > 0
                    sample_last_delete_pos = np.argwhere(
                        ((spike_xpos + afe.frame_right_windowsize) >= length_data_in) < 0).flatten()
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
                max_cluster_num = 0 if (first_run or cluster_class_avai) else (1 + np.argmax(np.unique(frames_cl)))
                if first_run:
                    endPoint = process_points[1] if use_end_point != 0 else datahandler._no_files
                    settings = afe.save_settings()
                    frames_in = frame_new
                    frames_cl = frame_cl + max_cluster_num
                else:
                    frames_in = np.concatenate((frames_in, frame_new), axis=0)
                    frames_cl = np.concatenate((frames_cl, frame_cl + max_cluster_num), axis=0)
                first_run = False

                if frames_in.shape[0] != frames_cl.size:
                    print(f'Data merging has an error after channel #{ch}')

                # --- Release memory
                del afe, spike_xpos, frame_new, frame_cl

            # --- Calculation of runtime duration
            time_stop = datetime.now()
            time_dt = time_stop - time_start
            output = np.unique(frames_cl, return_counts=True)
            print(f"... done after {time_dt.seconds + 1e-6 * time_dt.microseconds: .2f} s"
                  f" and recovered {ite_recoverd} samples")
            print(f"... available clusters: {output[0]} with samples: {output[1]}")

            # --- Saving data (each run)
            create_time = datetime.now().strftime("%Y-%m-%d")
            newfile_name = join(self.path2folder,
                                f"{create_time}_Dataset-{datahandler.raw_data.data_name}_step{runPoint + 1:03d}")
            savemat(f"{newfile_name}.mat",
                    {"frames_in": frames_in,
                     "frames_cl": frames_cl,
                     "create_time": create_time, "settings": settings},
                    do_compression=True, long_field_names=True)
            print(f'Saved file in: {newfile_name}.mat\n')

            # --- Release memory
            del datahandler, frames_in, frames_cl

            # --- End control routine
            runPoint += 1

        # --- The End
        print("... This is the end")

    def merge_data_from_diff_data(self):
        folder_content = glob(join(self.path2save, 'Merging', '*.mat'))
        folder_content.sort()

        frame_in = np.zeros((0, 0), dtype='int16')
        frame_cl = np.zeros((0, 0), dtype='int16')
        file_name = folder_content[-1].split('\\')[-1] if platform.system() == "Windows" else \
            folder_content[-1].split('/')[-1]
        file_name = file_name.split('_step')[0]

        for idx, file in enumerate(folder_content):
            data = loadmat(file)
            cl_in = data['frames_cl'] if 'frames_cl' in data else data['frames_cluster']
            frame_in = data['frames_in'] if idx == 0 else np.append(frame_in, data['frames_in'], axis=0)
            frame_cl = cl_in if idx == 0 else np.append(frame_cl, cl_in, axis=1)
            print(f"Read file: {file} and now cluster = {np.unique(frame_cl)}")

        # --- Transfering to mat-file
        frame_in = np.array(frame_in, dtype='int16')
        frame_cl = np.array(frame_cl, dtype='uint16')
        output = np.unique(frame_cl, return_counts=True)
        print(f"... available clusters: {output[0]} with samples: {output[1]}")
        self.filepath = join(self.path2save, file_name) + '_Merged.mat'
        savemat(self.filepath,
                {"frames_in": frame_in, "frames_cl": frame_cl,
                 "create_time": data['create_time'], "settings": data['settings']},
                do_compression=True, long_field_names=True)

        # --- Output of clustering
        num_clusters = np.unique(frame_cl, return_counts=True)
        print(f'Type of cluster classes: {num_clusters[0]}\n'
              f'Number of samples: {num_clusters[1]}')


def merge_frames_from_dataset() -> None:
    """Tool for merging all spike frames to one new dataset (Step 2)"""
    print("\nStart MATLAB script manually: merge/merge_datasets_matlab.m")
