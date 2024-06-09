import os
import numpy as np
import scipy.io
import numpy

from scipy.io import loadmat


class DataCompressor:

    def __init__(self, data_type):
        self.filepath = r"C:\Users\Haris\Documents\Master Sciebo neu\SS2024\CPS Projekt1\2024-02-05_Dataset-KlaesNeuralDecoding.mat"

        self.mode_data = data_type
        if data_type == 1:
            self.data_type = "Klaes-file"
            self.num_experiments = 21
            self.num_trials = 50
        elif data_type == 2:
            self.data_type = "Martinez-file"
        else:
            self.data_type = "not defined"

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

    def format_data(self):

        exp_index = 0
        trial_index = 0

        if self.data_type == "Klaes-file":

            loaded_data = self.load_Data()
            b = self.create_Dict()

            ##format data
            for exp_key in loaded_data:

                if exp_key.startswith("exp"):

                    exp_data = loaded_data[exp_key]

                    for trial_key in exp_data.dtype.names:
                        if trial_key.startswith("trial"):

                            trial_data = exp_data[trial_key][0, 0]
                            waveforms = trial_data["waveforms"][0, 0]

                            for electrode in range(96):
                                b[exp_index][trial_index][electrode] = waveforms[0][electrode]

                        trial_index += 1

                    trial_index = 0
                    exp_index += 1

            numpy_array = np.array(list(b.values()))  # cast dict as list -> np.array

            a = []

            waveform_index = 0
            for x in range(len(b)):
                for y in range(len(b[x])):
                    for z in range(len(b[x][y])):
                        for v in range(len(b[x][y][z])):
                            if len(b[x][y][z][v]) == 48:
                                # a.append([waveform_index]+ list(data[x][y][z][v])) # so steht der index vorne dran
                                a.append(b[x][y][z][v])
                                waveform_index += 1

                            else:
                                continue
            a = np.array(a)

            np.save("_waveforms_as_one_array" + ".npy", a)

            print(a.shape)


trialONE = DataCompressor(1)
trialONE.format_data()
