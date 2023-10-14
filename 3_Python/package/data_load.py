from os import listdir
from os.path import join, isdir
from glob import glob
import numpy as np
from scipy.io import loadmat


class DataHandler:
    """Class for datahandler"""
    # --- Meta Information
    data_name = None
    data_type = None
    gain = None
    noChannel = None
    # --- Data
    fs_orig = 0
    fs_used = 0
    channel = list()
    raw_data = list()
    # --- Behaviour
    behaviour_exist = False
    behaviour = None
    # --- GroundTruth
    label_exist = False
    spike_offset = list()
    spike_xpos = list()
    spike_no = list()
    cluster_id = list()
    cluster_no = list()


class DataLoader:
    """Class for loading and manipulating the used dataset"""
    def __init__(self) -> None:
        self.path2data = str()
        self.path2file = str()
        self.raw_data = DataHandler()

    def __prepare_access(self, folder_name: str, data_type: str, sel_datapoint: int) -> None:
        """Getting the file of the corresponding trial"""
        path = join(self.path2data, folder_name, data_type)
        folder_content = glob(path)
        folder_content.sort()
        self.no_files = len(folder_content)
        try:
            file_data = folder_content[sel_datapoint]
            self.path2file = join(self.path2data, folder_name, file_data)
        except:
            print("--- Folder not available - Please check folder name! ---")

    def __prepare_access_subfolder(self, folder_name: str, data_type: str, sel_dataset: int, sel_datapoint: int) -> None:
        """Getting the file structure within cases/experiments in one data set"""
        path2data = join(self.path2data, folder_name)
        path = join(path2data, data_type)
        folder_content = glob(path)
        folder_content.sort()
        folder_data = [name for name in listdir(path2data) if isdir(join(path2data, name))]
        file_data = folder_data[sel_dataset]

        path2data = join(path2data, file_data)
        self.__prepare_access(path2data, data_type, sel_datapoint)
        self.no_subfolder = len(file_data)

    def execute_data_call(self, data_type: int, data_set: int, data_point: int):
        """Loading the dataset"""
        if data_type == 1:
            self.__load_Martinez2009(data_set, data_point)
        elif data_type == 2:
            self.__load_Pedreira2012(data_set, data_point)
        elif data_type == 3:
            self.__load_Quiroga2020(data_set, data_point)
        elif data_type == 4:
            self.__load_Seidl2012(data_set, data_point)
        elif data_type == 5:
            self.__load_FZJ_MCS(data_set, data_point)
        elif data_type == 6:
            self.__load_KlaesLab(data_set, data_point)

    def __load_Martinez2009(self, case: int, point: int) -> None:
        """Loading synthethic files from Quiroga simulator (2009)"""
        self.__path2data = self.path2data
        folder_name = "01_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, point)

        loaded_data = loadmat(self.path2file)
        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = int(loaded_data["chan"][0])
        data.gain = 0.5e-6 * 10 ** (0 / 20)
        data.fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        data.raw_data = [(data.gain * loaded_data["data"][0])]
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = True
        data.spike_xpos = [(loaded_data["spike_times"][0][0][0])]
        data.cluster_id = [(loaded_data["spike_class"][0][0][0])]
        data.cluster_no = [np.unique(data.cluster_id[0]).size]
        data.spike_no = [data.spike_xpos[0].size]
        data.spike_offset = [100]
        # Return
        self.raw_data = data

    def __load_Pedreira2012(self, case: int, point: int) -> None:
        """Loading synthethic files from Quiroga simulator (2012)"""
        self.__path2data = self.path2data
        folder_name = "02_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, point)

        prep_index = self.path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = join(self.__path2data, folder_name, "ground_truth.mat")

        loaded_data = loadmat(self.path2file)
        ground_truth = loadmat(path2label)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = int(1)
        data.gain = 25e-6 * 10 ** (0 / 20)
        data.fs_orig = 24e3
        data.raw_data = [(data.gain * loaded_data["data"][0])]
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = True
        data.spike_xpos = [(ground_truth["spike_first_sample"][0][num_index - 1][0])]
        data.cluster_id = [(ground_truth["spike_classes"][0][num_index - 1][0])]
        data.cluster_no = [(np.unique(data.cluster_id[-1]).size)]
        data.spike_no = [(data.spike_xpos[-1].size)]
        data.spike_offset = [100]
        # Return
        self.raw_data = data

    def __load_Quiroga2020(self, case: int, point: int) -> None:
        """Loading synthetic recordings from Quiroga simulator (Common benchmark)"""
        self.__path2data = self.path2data
        folder_name = "03_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, point)

        loaded_data = loadmat(self.path2file, mat_dtype=True)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = int(1)
        data.gain = 100e-6 * 10 ** (0 / 20)
        data.fs_orig = float(1000 / loaded_data["samplingInterval"][0][0])
        data.raw_data = [(data.gain * loaded_data["data"][0])]
        # Behaviour
        data.behaviour_exist = False
        data.behaviour = None
        # Groundtruth
        data.label_exist = True
        data.spike_xpos = [(loaded_data["spike_times"][0][0][0])]
        data.cluster_id = [(loaded_data["spike_class"][0][0][0]-1)]
        data.cluster_no = [(np.unique(data.cluster_id[-1]).size)]
        data.spike_no = [(data.spike_xpos[-1].size)]
        data.spike_offset = [500]
        # Return
        self.raw_data = data

    def __load_Seidl2012(self, case: int, point: int) -> None:
        """Loading the recording files from the Freiburg probes from Karsten Seidl from this PhD"""
        self.__path2data = self.path2data
        folder_name = "04_Freiburg_Seidl2014"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, point)

        loaded_data = loadmat(self.path2file)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Penetrating"
        data.noChannel = loaded_data['noChannel'][0][0]
        data.gain = loaded_data['GainPre'][0][0]
        data.fs_orig = loaded_data['origFs'][0][0]

        raw = loaded_data['raw_data']
        for idx in range(0, data.noChannel):
            data.raw_data.append(raw[idx, :]/data.gain)
        # Behaviour
        data.behaviour_exist = False
        data.behaviour = None
        # Groundtruth
        data.label_exist = False
        data.spike_offset = [0]
        # Return
        self.raw_data = data

    def __load_FZJ_MCS(self, case: int, point: int) -> None:
        """Loading the recording files from MCS setup in FZ Juelich (case = experiment, point = file)"""
        self.__path2data = self.path2data
        folder_name = "05_FZJ_MCS"
        data_type = '*_new.mat'
        self.__prepare_access(folder_name, data_type, point)

        loaded_data = loadmat(self.path2file)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "MCS 60MEA"
        data.gain = loaded_data['gain'][0]
        data.fs_orig = 1e3 * loaded_data['fs'][0]

        raw = loaded_data['raw']/data.gain
        data.noChannel = raw.shape[1]
        for idx in range(0, data.noChannel):
            data.raw_data.append(raw[:, idx])

        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = False
        data.spike_offset = [0]
        # Return
        self.raw_data = data

    def __load_KlaesLab(self, case: int, nsp_device: int) -> None:
        """Loading the *.ns6 and *.nev files from recordings with Utah array from Blackrock Neurotechnology (case = experiment, nsp_device)"""
        self.__path2data = self.path2data
        folder_name = "10_Klaes_Caltech"
        data_type = '*_MERGED.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, nsp_device)

        loaded_data = loadmat(self.path2file, mat_dtype=True)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Utah"
        data.noChannel = int(loaded_data['rawdata']['NoElectrodes'][0, 0][0])
        data.gain = 0.25e-6
        # data.gain = loaded_data['rawdata']['LSB'][0, 0][0]
        data.fs_orig = int(loaded_data['rawdata']['SamplingRate'][0, 0][0])

        raw = data.gain * loaded_data['rawdata']['spike'][0, 0]
        for idx in range(0, data.noChannel):
            data.raw_data.append(raw[:, idx])

        # --- Behaviour
        data.behaviour_exist = True
        data.behaviour = loaded_data['behaviour']
        # --- Groundtruth from BlackRock
        data.label_exist = int(loaded_data['nev_detected']['Exits'][0, 0][0])
        # Processing of electrode information
        nev_waveform = list()
        for idx in range(1, data.noChannel+1):
            str_out = 'Elec'+ str(idx)
            A = (loaded_data['nev_detected'][str_out][0, 0]['timestamps'][0, 0][0, :])
            B = (loaded_data['nev_detected'][str_out][0, 0]['cluster'][0, 0][0, :])
            C = (loaded_data['nev_detected'][str_out][0, 0]['waveform'][0, 0])
            D = len(A)
            data.spike_xpos.append(A)
            data.spike_no.append(D)
            data.cluster_id.append(B)
            data.cluster_no.append(np.unique(B).size)
            data.spike_offset.append(100)
            nev_waveform.append(C)
        # Return
        self.raw_data = data
        self.nev_waveform = nev_waveform
