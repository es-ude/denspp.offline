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



    def get_Path(self, data_type):
        current_dir = os.getcwd()
        target_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'data')
        os.makedirs(target_path, exist_ok=True)
        file_path = os.path.join(target_path, f"{data_type}_as_one_array.npy")
        return file_path

    def create_Dict(self):
        # easier to use a dict, since some electrodes have more waveforms than others
        b = {}
        for x in range(self.num_experiments):  # 21 experiments
            b[x] = {}
            for y in range(self.num_trials):  # 50 trials
                b[x][y] = {}
        return b

    def extract_from_Klaes(self, feature):
        loaded_data = loadmat(self.filepath)
        exp_index = 0
        trial_index = 0

        b = self.create_Dict()
        position = []
        counter = 1
        for exp_key in loaded_data:
            if exp_key.startswith("exp"):
                exp_data = loaded_data[exp_key]

                for trial_key in exp_data.dtype.names:
                    if trial_key.startswith("trial"):
                        waveforms = exp_data[trial_key][0, 0][feature][0, 0]

                        for electrode in range(len(waveforms[0])):
                            b[exp_index][trial_index][electrode] = waveforms[0][electrode]
                            for x in range(len(waveforms[0][electrode])):
                                position.append(electrode)


                    trial_index += 1
                trial_index = 0
                exp_index += 1

        return b, position

    def __is_valid(self, waveform):
        lowestindex_current = np.argmin(waveform)
        highestindex_current = np.argmax(waveform)
        if ((lowestindex_current == 11 or lowestindex_current == 12 or lowestindex_current == 13) and
                (highestindex_current < 1 or highestindex_current > 13) and (highestindex_current < 40)):
            return True
        else:
            return False

    def iterate_over_dataset(self, dataset):
        a = []
        lowestindex = []
        highestindex = []
        for exp in dataset:
            for trial in dataset[exp]:
                for electrode in dataset[exp][trial]:
                    for waveforms in dataset[exp][trial][electrode]:

                        #if self.feature == "waveforms" and self.__is_valid(waveforms):
                            a.append(waveforms)
                            highestindex.append(np.argmax(waveforms))
                            lowestindex.append(np.argmin(waveforms))


        return a
    ### main function
    def format_data(self):

        valid_waveforms = []
        valid_timestamps = []
        valid_positions = []

        ####DIE FOLGENDEN ZWEI ZEILEN NICHT IN REIHENFOLGE VERTAUSCHEN!!!ELF
        timestamp_dataset, position = self.extract_from_Klaes("timestamps")
        dataset, position = self.extract_from_Klaes("waveforms")




        ### format to one 2D array
        waveforms_as_array = self.iterate_over_dataset(dataset)
        timestamps_as_array = self.iterate_over_dataset(timestamp_dataset)

        timestamps_as_array = np.concatenate(timestamps_as_array)
        waveforms_as_array = np.array(waveforms_as_array)


        for x in range(len(waveforms_as_array)):
            if self.__is_valid(waveforms_as_array[x]):
                valid_waveforms.append(waveforms_as_array[x])
                valid_timestamps.append(timestamps_as_array[x])
                valid_positions.append(position[x])


        valid_timestamps = np.array(valid_timestamps)
        valid_waveforms = np.array(valid_waveforms)
        valid_positions = np.array(valid_positions)

        print(valid_positions.shape)
        print(valid_waveforms.shape)
        print(valid_timestamps.shape)

        np.save(self.get_Path("waveforms"), valid_waveforms)
        np.save(self.get_Path("timestamps"), valid_timestamps)
        np.save(self.get_Path("positions"), valid_positions)


trialONE = DataCompressor()
trialONE.format_data()
data = np.load(trialONE.get_Path("waveforms"))
print(data[3])

#print(data["valid_waveforms"][3])
#print(data["valid_positions"][3])
