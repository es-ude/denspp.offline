import os
import numpy as np
import scipy.io
import numpy

from scipy.io import loadmat

class DataCompressor:

    def __init__(self, data_type):
        self.filepath= r"C:\Users\Haris\Documents\Master Sciebo neu\SS2024\CPS Projekt1\2024-02-05_Dataset-KlaesNeuralDecoding.mat"



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

    def format_data_klaes(self):
        trial_index = 0
        trial_index = 0
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
                            b[trial_index][trial_index][electrode] = waveforms[0][electrode]

                    trial_index += 1

                trial_index = 0
                trial_index += 1

        numpy_array = np.array(list(b.values()))  # cast dict as list -> np.array

        numpy.save("_waveforms" + ".npy", numpy_array)




