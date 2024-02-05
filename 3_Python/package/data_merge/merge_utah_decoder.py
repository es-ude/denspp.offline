import os
from datetime import datetime
import numpy
import pandas as pd
import scipy.io
from numpy import size
import numpy as np
from scipy.io import loadmat


class DatasetDecoder:

    def __init__(self, path2save: str):
        self.folder_name = "06_Klaes_Caltech"
        self.data_type = '-NSP1-001_MERGED.mat'
        self.path2save = path2save
        self._num_channels = 96

        create_time = datetime.now().strftime("%Y-%m-%d")
        self.name_for_saving = f"{create_time}_Dataset-KlaesNeuralDecoding"

    def generateDataset(self, path2folder: str) -> None:
        """Function for Generating Training from Human Brain Recordings (by using Utah array)

        Args:
            path2folder: Specific place in which the data is stored
        """
        path2experiments = os.path.join(path2folder, self.folder_name)

        # --- Generate folder for saving datasets
        if not os.path.exists(self.path2save):
            os.mkdir(self.path2save)

        # --- Loading the electrode mapping
        path2map = os.path.join(path2experiments, 'mapping.csv')

        # --- Processing data
        all_data = dict()
        cnt_exp = 0
        for exp_folder in os.listdir(path2experiments):
            path2folder = os.path.join(path2experiments, exp_folder)
            if os.path.isdir(path2folder):
                for nsp_file in os.listdir(path2folder):
                    if nsp_file.endswith(self.data_type):
                        path2nsp = os.path.join(path2experiments, exp_folder, nsp_file)
                        loaded_mat_file = self.__load_matfile(path2nsp)

                        # --- Processing specific file from Utah recordings
                        data_nev = loaded_mat_file['nev_detected'][0, 0]
                        data_beh = loaded_mat_file['behaviour'][0, 0]['saveData'][0, 0]
                        data_raw = loaded_mat_file['rawdata'][0, 0]

                        trials_data = self.__process_experiment_data(data_nev, data_raw, data_beh)
                        trials_data.update({'name': str(exp_folder + '/' + nsp_file)})
                        trials_data.update({'orientation': self.load_electrode_orientation(path2map)})

                        # -- Return
                        all_data.update({f'exp_{cnt_exp:03d}': trials_data})
                        cnt_exp += 1
                        print(f'... processed file {cnt_exp}: {path2nsp}')

        # --- Saving data
        print('... saving data')
        self.__save_experiment_data(all_data)

    def __load_matfile(self, nsp):
        mat_data_path = os.path.join(os.getcwd(), nsp)
        mat_data = loadmat(mat_data_path)

        return mat_data

    def __process_experiment_data(self, data_nev, data_raw, data_beh):
        trials_data = {}
        fs = int(data_raw['SamplingRate'][0, 0])

        # --- Processing data
        num_trials = size(data_beh['Trials'])
        for ite in range(0, num_trials):
            trials_data[f'trial_{ite:03d}'] = self.__process_trial_data(data_nev, data_beh, ite, fs)

        return trials_data

    def __process_trial_data(self, loaded_nev_file, loaded_behaviour_file, trial_number: int, sampling_rate: int) -> dict:
        # --- Preparing data structure
        data_event_used = [[] for _ in range(0, self._num_channels)]
        data_event_orig = [[] for _ in range(0, self._num_channels)]
        data_cluster = [[] for _ in range(0, self._num_channels)]
        data_waveform = [[] for _ in range(0, self._num_channels)]
        data_behav = {'decision': loaded_behaviour_file['Trials']['Effector'][0, trial_number][0],
                      'exp_says': loaded_behaviour_file['Trials']['ButtonPressed'][0, trial_number][0],
                      'patient_says': loaded_behaviour_file['Trials']['ActionType'][0, trial_number][0]}

        # --- Loading data from experiment
        for electrode in loaded_nev_file.dtype.names:
            if "Elec" in electrode:
                # --- Getting the data
                timestamps = loaded_nev_file[electrode][0, 0]['timestamps'].flatten()
                cluster = loaded_nev_file[electrode][0, 0]['cluster'].flatten()
                waveform = loaded_nev_file[electrode][0, 0]['waveform']
                #elec_num = electrode.lower()

                # --- Cutting region of interest
                trial_start_time = loaded_behaviour_file['EventTimes'][trial_number, 0] - 1
                trial_end_time = loaded_behaviour_file['EventTimes'][trial_number, 5] + 1

                relevant_indices = ((timestamps / sampling_rate >= trial_start_time) &
                                    (timestamps / sampling_rate <= trial_end_time))
                idx_true = np.argwhere(relevant_indices.flatten() == True).flatten()

                # --- Transfer to output
                if not idx_true.size == 0:
                    timestamps_orig = timestamps[idx_true].tolist()
                    timestamps_used = (timestamps[idx_true] - int(trial_start_time * sampling_rate)).tolist()
                    cluster_used = cluster[idx_true].tolist()
                    waveform_used = waveform[idx_true].tolist()
                else:
                    timestamps_orig = []
                    timestamps_used = []
                    cluster_used = []
                    waveform_used = []

                num_electrode = int(electrode.split('Elec')[1]) - 1
                data_event_orig[num_electrode] = timestamps_orig
                data_event_used[num_electrode] = timestamps_used
                data_cluster[num_electrode] = cluster_used
                data_waveform[num_electrode] = waveform_used

        data_out = {
            'timestamps': data_event_used,
            'timestamps_raw': data_event_orig,
            'cluster': data_cluster,
            'waveforms': data_waveform,
            'label': data_behav,
            'samplingrate': sampling_rate
        }

        return data_out

    def load_electrode_orientation(self, path2filecsv: str):
        electrodeOrientation = pd.read_csv(path2filecsv, sep=';')
        return electrodeOrientation

    def __save_experiment_data(self, mdic: dict) -> None:
        path2save = os.path.join(self.path2save, self.name_for_saving)

        scipy.io.savemat(path2save + '.mat', mdic)
        numpy.save(path2save + '.npy', mdic)
