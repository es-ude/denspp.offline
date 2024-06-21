import os
import numpy as np
from scipy.io import loadmat


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


class DataCompressor:

    def __init__(self):

            self.filepath = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data",
                                         "2024-02-05_Dataset-KlaesNeuralDecoding.mat")
            self.num_experiments = 21
            self.num_trials = 50

    def get_Path(self):
        current_dir = os.getcwd()
        target_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')
        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, f"waveforms_as_one_array.npy")
        return file_path

    def create_Dict(self):
        # easier to use a dict, since some electrodes have more waveforms than others
        b = {}
        for x in range(self.num_experiments):  # 21 experiments
            b[x] = {}
            for y in range(self.num_trials):  # 50 trials
                b[x][y] = {}
        return b

    def extract_from_Klaes(self):
        loaded_data = loadmat(self.filepath)
        exp_index = 0
        trial_index = 0
        b = self.create_Dict()

        for exp_key in loaded_data:
            if exp_key.startswith("exp"):
                exp_data = loaded_data[exp_key]

                for trial_key in exp_data.dtype.names:
                    if trial_key.startswith("trial"):
                        waveforms = exp_data[trial_key][0, 0]["waveforms"][0, 0]

                        for electrode in range(len(waveforms[0])):
                            b[exp_index][trial_index][electrode] = waveforms[0][electrode]

                    trial_index += 1
                trial_index = 0
                exp_index += 1
        return b

    def __is_valid(self, waveform):
        lowestindex_current = np.argmin(waveform)
        highestindex_current = np.argmax(waveform)
        if ((lowestindex_current == 11 or lowestindex_current == 12 or lowestindex_current == 13) and
                (highestindex_current < 1 or highestindex_current > 13) and (highestindex_current < 40)):
            return True
        else:
            return False


    ### main function
    def format_data(self):


        b = self.extract_from_Klaes()

        ### format to one 2D array
        a = []
        lowestindex = []
        highestindex = []
        waveform_index = 0
        for exp in b:
            for trial in b[exp]:
                for electrode in b[exp][trial]:
                    #if feature == "waveforms":
                    for waveforms in b[exp][trial][electrode]:
                        # a.append([waveform_index]+ list(data[x][y][z][v])) # so steht der index vorne dran #waveform index muss inkrementiert werden

                        if self.__is_valid(waveforms):
                            a.append(normalize(waveforms, 0 ,1 ))
                            highestindex.append(np.argmax(waveforms))
                            lowestindex.append(np.argmin(waveforms))


        a = np.array(a)

        np.save(self.get_Path(), a)
        #np.save(self.get_Path(), timestamp)

        print(a.shape)
        print(a[0:2])


trialONE = DataCompressor()
trialONE.format_data()
b = np.load(trialONE.get_Path())
#print(b[3])
