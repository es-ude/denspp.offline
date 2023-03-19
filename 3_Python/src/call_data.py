import os, glob
import numpy as np
from typing import Tuple
from fractions import Fraction
from scipy.io import loadmat
from scipy.signal import resample_poly

# ----- Read Settings -----
class DataHandler():
    # --- Meta Information
    data_name = None
    data_type = None
    gain = None
    noChannel = None
    # --- Data
    fs_orig = None
    fs_used = None
    raw_data = None
    # --- GroundTruth
    label_exist = False
    spike_xpos = None
    spike_no = 0
    cluster_id = None
    cluster_no = 0

# TODO: Mehrkanal-Einlesung und Auswertung einfügen
class DataController(DataHandler):
    def __init__(self, path2data: str, sel_channel: int):
        # Meta-Information about datasets
        self.max_datapoints = np.array([5, 16, 22, 2])
        # Settings
        self.raw_data = DataHandler()
        self.path2data = path2data
        self.path2file = None
        # if sel_channel = 0 -> All data
        self.used_channel = sel_channel
        self.__fill_factor = 0
        self.__scaling = 1


    def do_call(self, data_type: int, data_set: int):
        # ----- Read data input -----
        if data_type == 1:
            self.__load_Martinez2009(data_set)
        elif data_type == 2:
            self.__load_Pedreira2012(data_set)
        elif data_type == 3:
            self.__load_Quiroga2020(data_set)
        elif data_type == 4:
            self.__load_Seidl2012(data_set)
        elif data_type == 5:
            self.__load_Klaes2016(data_set)

        self.raw_data.fs_used = self.raw_data.fs_orig
        print("... using data point:", self.path2file)

    def do_resample(self, t_range: np.array, desired_fs: int):
        self.raw_data.fs_used = desired_fs

        do_resampling = bool(self.raw_data.fs_used != self.raw_data.fs_orig)
        data_in = self.raw_data.raw_data[self.used_channel]
        spikepos_in = self.raw_data.spike_xpos
        cluster_in = self.raw_data.cluster_id

        # --- Cutting specific time range out of raw data
        if t_range.size == 2:
            idx0 = int(t_range[0] * self.raw_data.fs_orig)
            idx1 = int(t_range[1] * self.raw_data.fs_orig)

            data_in = data_in[idx0:idx1]
            self.__fill_factor = (idx0 - idx1) / data_in.size

        # --- Resampling the input
        if do_resampling:
            u_safe = 5e-6
            u_chck = np.mean(data_in[0:10])
            if np.abs((u_chck < u_safe) - 1) == 1:
                du = u_chck
            else:
                du = 0

            (p, q) = self.__get_resample_ratio(self.raw_data.fs_orig, self.raw_data.fs_used)

            self.raw_data.raw_data = du + resample_poly(data_in - du, p, q)
            self.__scaling = p / q
        else:
            self.raw_data.raw_data = data_in
            self.__scaling = 1

        # --- "Resampling" the labeled informations
        if self.raw_data.label_exist:
            if t_range.size == 2:
                # Find values from x-positions
                idx2 = np.argwhere(spikepos_in >= idx0)
                idx2 = 1+int(idx2[0])
                idx3 = np.argwhere(spikepos_in <= idx1)
                idx3 = int(idx3[-1])
            else:
                idx2 = 0
                idx3 = -1

            self.raw_data.cluster_id = cluster_in[idx2:idx3]
            self.raw_data.spike_xpos = self.__scaling * spikepos_in[idx2:idx3]
            self.raw_data.cluster_no = np.unique(self.raw_data.cluster_id).size
            self.raw_data.spike_no = self.raw_data.spike_xpos.size

    # ---- Output of meta informations
    def output_meta(self):
        print(f"... using data set of: {self.raw_data.data_name}")
        print(f"... original sampling rate of {int(1e-3 * self.raw_data.fs_orig)} kHz (resampling to {int(1e-3 * self.raw_data.fs_used)} kHz)")
        print(f"... using", round(self.__fill_factor * 100 , 2), "% of the data (time length of", round(self.raw_data.raw_data.size / self.raw_data.fs_used / self.fill_factor, 2), "s)")
        print(f"... data includes", self.raw_data.noChannel, "number of electrode (" + self.raw_data.data_type)
        if self.raw_data.label_exist:
            print(f"... includes labels (noSpikes: {self.raw_data.spike_no} - noCluster: {self.raw_data.cluster_no}")
        else:
            print(f"... excludes labels")

    def get_data(self) -> DataHandler:
        return self.raw_data

    def __get_resample_ratio(self, fin: int, fout: int) -> Tuple[int, int]:
        calced_fs = fout / fin
        (p, q) = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()
        return (p, q)

    def __prepare_access(self, folder_name: str, data_type: str, sel_dataset: int) -> None:
        folder_content = glob.glob(os.path.join(self.path2data, folder_name, data_type))
        folder_content.sort()
        file_data = folder_content[sel_dataset - 1]
        self.path2file = os.path.join(self.path2data, folder_name, file_data)


    def __load_Martinez2009(self, indices: list = np.arange(1, 5)) -> None:
        folder_name = "01_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        loaded_data = loadmat(self.path2file)

        # Data
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.noChannel = int(loaded_data["chan"][0])
        self.raw_data.gain = 10 ** (0 / 20)
        self.raw_data.fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        self.raw_data.raw_data = 0.5e-6 * loaded_data["data"]
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_xpos = loaded_data["spike_times"][0][0][0]
        self.raw_data.cluster_id = loaded_data["spike_class"][0][0][0]
        self.raw_data.cluster_no = np.unique(self.raw_data.cluster_id).size
        self.raw_data.spike_no = self.raw_data.spike_xpos.size

    def __load_Pedreira2012(self, indices: list = np.arange(1, 16)) -> None:
        folder_name = "02_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        prep_index = self.path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = os.path.join(self.path2data, folder_name, "ground_truth.mat")

        loaded_data = loadmat(self.path2file)
        ground_truth = loadmat(path2label)

        # Data
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.noChannel = int(1)
        self.raw_data.gain = 10 ** (0 / 20)
        self.raw_data.fs_orig = int(24000)
        self.raw_data.raw_data = 25e-6 * loaded_data["data"]
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_xpos = ground_truth["spike_first_sample"][0][num_index - 1][0]
        self.raw_data.cluster_id = ground_truth["spike_classes"][0][num_index - 1][0]
        self.raw_data.cluster_no = np.unique(self.raw_data.cluster_id).size
        self.raw_data.spike_no = self.raw_data.spike_xpos.size


    def __load_Quiroga2020(self, indices: list = np.arange(1, 22)) -> None:
        folder_name = "03_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        loaded_data = loadmat(self.path2file)

        # Data
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.noChannel = int(1)
        self.raw_data.gain = 10 ** (0 / 20)
        self.raw_data.fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        self.raw_data.raw_data = 100e-6 * loaded_data["data"]
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_xpos = loaded_data["spike_times"][0][0][0]
        self.raw_data.cluster_id = loaded_data["spike_class"][0][0][0]-1
        self.raw_data.cluster_no = np.unique(self.raw_data.cluster_id).size
        self.raw_data.spike_no = self.raw_data.spike_xpos.size

    def __load_Seidl2012(self, indices: list = np.arange(1,2)) -> None:
        folder_name = "04_Freiburg_Seidl2014"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        loaded_data = loadmat(self.path2file)

        self.raw_data.data_name = folder_name
        self.raw_data.data_type = loaded_data["TypeMEA"][0][0]
        self.raw_data.noChannel = loaded_data['noChannel'][0][0]
        self.raw_data.gain = loaded_data['GainPre'][0][0]
        self.raw_data.fs_orig = loaded_data['origFs'][0][0]
        self.raw_data.raw_data = loaded_data['raw_data']
        # Groundtruth
        self.raw_data.label_exist = False
        self.raw_data.spike_xpos = None
        self.raw_data.cluster_id = None
        self.raw_data.cluster_no = 0
        self.raw_data.spike_no = 0

    # TODO: Andere Quellen noch anpassen
    def __load_Klaes2016(self, indices: list = np.arange(1, 2)) -> None:
        folder_name = "05_Daten_Klaes"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        # Alter Code
        """ 
            sessions = list(set(os.listdir(path2folder)).difference(set([".DS_Store", ".", ".."])))
            expected_subfolders = ["NSP1", "NSP2"]
            
            for session in sessions:
            for subfolder in expected_subfolders:
                path = os.sep.join([folder, session, subfolder])
                files = os.listdir(path)
                recordings = [x for x in files if ".mat" in x]
                for record in recordings:
                    if "102124-NSP1" in record:
                        path_data = os.sep.join([path, record])
                        path_metadata = path_data.replace(".mat", ".ccf")
                        data = mat73.loadmat(path_data)
        """
