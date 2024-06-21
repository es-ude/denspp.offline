import numpy as np

from scipy.io import loadmat


class DataCompressor:

    def __init__(self, data_type):

        self.mode_data = data_type
        if data_type == 1:
            self.data_type = "Klaes-file"
            self.filepath = r"C:\Users\Haris\Documents\Master Sciebo neu\SS2024\CPS Projekt1\old\2024-02-05_Dataset-KlaesNeuralDecoding.mat"
            self.num_experiments = 21
            self.num_trials = 50
        elif data_type == 2:
            self.data_type = "Martinez-file"
        else:
            self.data_type = "not defined"

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

    def format_data(self):

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
                            waveforms = trial_data["waveforms"][0, 0]
                            for electrode in range(96):
                                b[exp_index][trial_index][electrode] = waveforms[0][electrode]
                        trial_index += 1
                    trial_index = 0
                    exp_index += 1

            ### format to one 2D array
            a = []
            lowestindex = []
            highestindex =[]
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
                                    if highestindex_current < 1 or highestindex_current >13:
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
trialONE.format_data()
a = np.load(
    r"C:\Users\Haris\git\CPS_Projekt\denspp.offline\3_Python\package\data_merge\_waveforms_as_one_array_cleaned.npy")
b = np.load(r"C:\Users\Haris\git\CPS_Projekt\denspp.offline\3_Python\package\data_merge\lowest_index_of_waveforms.npy")
c = np.load(r"C:\Users\Haris\git\CPS_Projekt\denspp.offline\3_Python\package\data_merge\highest_index_of_waveforms.npy")
print(b)
counts = np.bincount(c)
print(counts)
for number, count in enumerate(counts):
    print(f"Zahl {number} kommt {count / 783830} Mal vor.")
