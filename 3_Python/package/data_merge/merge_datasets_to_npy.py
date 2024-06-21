import os

import numpy as np

from scipy.io import loadmat


class DataCompressor:

    def __init__(self, data_type):


        self.mode_data = data_type
        if data_type == 1:
            self.data_type = "Klaes-file"
            self.filepath = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data",
                                         "2024-02-05_Dataset-KlaesNeuralDecoding (2).mat")
            self.num_experiments = 21
            self.num_trials = 50
        elif data_type == 2:
            self.data_type = "Martinez-file"
        else:
            self.data_type = "not defined"

            raise ValueError("Not defined")

    def normalize(self, arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        diff_arr = max(arr) - min(arr)
        for i in arr:
            temp = (((i - min(arr)) * diff) / diff_arr) + t_min
            norm_arr.append(temp)
        return norm_arr

    def create_Dict(self):
        # easier to use a dict, so the trial/exp-keys can be the indices
        # otherwise you would need to cast every key as int
        b = {}
        for x in range(self.num_experiments):  # 21 experiments
            b[x] = {}
            for y in range(self.num_trials):  # 50 trials
                b[x][y] = {}
        return b

    def load_Data(self):
        loaded_data = loadmat(self.filepath)
        return loaded_data

    def get_Path(self, feature):
        current_dir = os.getcwd()
        target_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')
        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, f"waveforms_as_one_array_{feature}.npy")
        return file_path

    def format_data(self, feature):

        exp_index = 0
        trial_index = 0

        if self.data_type == "Klaes-file":

            loaded_data = self.load_Data()
            b = self.create_Dict()

            ##format matlab data to 4D Array
            for exp_key in loaded_data:
                if exp_key.startswith("exp"):
                    exp_data = loaded_data[exp_key]
                    for trial_key in exp_data.dtype.names:
                        if trial_key.startswith("trial"):
                            trial_data = exp_data[trial_key][0, 0]
                            waveforms = trial_data[feature][0, 0]                                                       # change from "waveforms" to feature to extract different parts
                            for electrode in range(96):
                                b[exp_index][trial_index][electrode] = waveforms[0][electrode]
                        trial_index += 1
                    trial_index = 0
                    exp_index += 1


            ### format to one 2D array
            a = []
            lowestindex = []
            highestindex = []
            waveform_index = 0
            for x in range(len(b)):
                for y in range(len(b[x])):
                    for z in range(len(b[x][y])):
                        for v in range(len(b[x][y][z])):
                            if len(b[x][y][z][v]) == 48:
                                # a.append([waveform_index]+ list(data[x][y][z][v])) # so steht der index vorne dran
                                # a.append(self.normalize(b[x][y][z][v], 0,1))        # so normalisiert, schlecht für kmeans

                                lowestindex_current = np.argmin(b[x][y][z][v])
                                highestindex_current = np.argmax(b[x][y][z][v])
                                if lowestindex_current == 11 or lowestindex_current == 12 or lowestindex_current == 13:
                                    if highestindex_current < 1 or highestindex_current > 13:
                                        a.append(b[x][y][z][v])
                                        highestindex.append(highestindex_current)
                                        lowestindex.append(lowestindex_current)
                                        waveform_index += 1

                            else:
                                continue
            a = np.array(a)

            np.save("_waveforms_as_one_array_cleaned" + ".npy", a)
            np.save("lowest_index_of_waveforms" + ".npy", lowestindex)
            np.save("highest_index_of_waveforms" + ".npy", highestindex)

            print(a.shape)


trialONE = DataCompressor(1)
trialONE.format_data("timestamps")
trialONE.format_data("waveforms")
a = np.load(trialONE.get_Path("timestamps"))
b = np.load(trialONE.get_Path("waveforms"))
print(a[3])
print(b[3])
