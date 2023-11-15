from os import listdir
from os.path import join, isdir
from glob import glob
import numpy as np
from scipy.io import loadmat
from mat73 import loadmat as loadmat_mat73
from package.data.data_addon import RGC_Cell_Names


# TODO: Device Selection is still missing
class DataHandler:
    """Class with data and meta information of the used neural dataset"""
    # --- Meta Information
    data_name = str()
    data_type = str()
    data_fs_orig = 0
    data_fs_used = 0
    data_lsb = 1.0
    data_time = 0.0
    # Num of devices
    device_id = str()
    # Num of electrodes per device
    electrode_id = list()
    # --- Data
    data_raw = list()

    # --- GroundTruth: Spike Sorting (per Channel)
    label_exist = False
    spike_offset_us = list()
    spike_ovrlap = list()
    spike_xpos = list()
    cluster_id = list()
    # --- Behaviour (in total of MEA)
    behaviour_exist = False
    behaviour = None


class DataLoader:
    """Class for loading and manipulating the used dataset"""
    def __init__(self) -> None:
        self.select_electrodes = list()
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

    def __prepare_access_subfolder(self, folder_name: str, data_type: str,
                                   sel_dataset: int, sel_datapoint: int) -> None:
        """Getting the file structure within cases/experiments in one data set"""
        path2data = join(self.path2data, folder_name)
        folder_data = [name for name in listdir(path2data) if isdir(join(path2data, name))]
        folder_data.sort()
        file_data = folder_data[sel_dataset]

        path2data = join(path2data, file_data)
        self.__prepare_access(path2data, data_type, sel_datapoint)
        self.no_subfolder = len(file_data)

    def execute_data_call(self, data_type: int, data_set: int, data_point: int):
        """Loading the dataset"""
        if data_type == 1:
            self.__load_martinez2009(data_set, data_point)
        elif data_type == 2:
            self.__load_pedreira2012(data_set, data_point)
        elif data_type == 3:
            self.__load_quiroga2020(data_set, data_point)
        elif data_type == 4:
            self.__load_seidl2012(data_set, data_point)
        elif data_type == 5:
            self.__load_marre2018(data_set, data_point)
        elif data_type == 6:
            self.__load_klaes_utah_array(data_set, data_point)
        elif data_type == 7:
            self.__load_rgc_tdb(data_set, data_point)
        elif data_type == 8:
            self.__load_fzj_mcs(data_set, data_point)
        elif data_type == 9:
            self.__load_musall_neuropixel(data_set, data_point)
        else:
            print("\nPlease select new input for data_type! -> [1, 9]")

    def __load_martinez2009(self, case: int, point: int) -> None:
        """Loading synthethic files from Quiroga simulation (2009)"""
        self.__path2data = self.path2data
        folder_name = "01_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, point)

        loaded_data = loadmat(self.path2file)
        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_lsb = 0.5e-6
        self.raw_data.data_fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)

        self.raw_data.device_id = [0]
        self.raw_data.electrode_id = [int(loaded_data["chan"][0])-1]
        self.raw_data.data_raw = [self.raw_data.data_lsb * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"][0].size / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_offset_us = [-100]
        self.raw_data.spike_ovrlap = list()
        self.raw_data.spike_xpos = [(loaded_data["spike_times"][0][0][0])]
        self.raw_data.cluster_id = [(loaded_data["spike_class"][0][0][0])]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_pedreira2012(self, case: int, point: int) -> None:
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

        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_lsb = 25e-6
        self.raw_data.data_fs_orig = 24e3

        self.raw_data.device_id = [0]
        self.raw_data.electrode_id = [int(loaded_data["data"].shape[0])-1]
        self.raw_data.data_raw = [self.raw_data.data_lsb * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"].shape[1] / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_offset_us = [-100]
        self.raw_data.spike_ovrlap = list()
        self.raw_data.spike_xpos = [(ground_truth["spike_first_sample"][0][num_index - 1][0])]
        self.raw_data.cluster_id = [(ground_truth["spike_classes"][0][num_index - 1][0])]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_quiroga2020(self, case: int, point: int) -> None:
        """Loading synthetic recordings from Quiroga simulator (Common benchmark)"""
        self.__path2data = self.path2data
        folder_name = "03_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, point)
        loaded_data = loadmat(self.path2file, mat_dtype=True)

        # --- Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_lsb = 100e-6
        self.raw_data.data_fs_orig = float(1000 / loaded_data["samplingInterval"][0][0])

        self.raw_data.device_id = [0]
        self.raw_data.electrode_id = [int(loaded_data["chan"][0][0])-1]
        self.raw_data.data_raw = [self.raw_data.data_lsb * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"].shape[1] / self.raw_data.data_fs_orig
        # --- Groundtruth
        self.raw_data.label_exist = True
        self.raw_data.spike_offset_us = [-500]
        # Process overlapping data
        ovrlap_input = np.array(loaded_data["OVERLAP_DATA"][0], dtype=int)
        ovrlap_event_pos = np.where(ovrlap_input != 0)[0]
        sel_pos = np.where(np.diff(ovrlap_event_pos) != 1)[0]
        ovrlap_event_pos0 = np.append(0, 1 + sel_pos[0:-1])
        ovrlap_event_pos1 = sel_pos
        ovrlap_event_pos2 = ovrlap_event_pos[ovrlap_event_pos0]
        ovrlap_event_pos3 = ovrlap_event_pos[ovrlap_event_pos1]
        for idx, pos in enumerate(ovrlap_event_pos2):
            pos_strt = pos
            pos_ends = ovrlap_event_pos3[idx]
            sel_id = ovrlap_input[pos_strt:pos_ends]
            self.raw_data.spike_ovrlap.append([np.array((pos_strt, pos_ends), dtype=int), np.unique(sel_id)])

        self.raw_data.spike_xpos = [(loaded_data["spike_times"][0][0][0])]
        self.raw_data.cluster_id = [(loaded_data["spike_class"][0][0][0]-1)]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_seidl2012(self, case: int, point: int) -> None:
        """Loading the recording files from the Freiburg probes from Karsten Seidl from this PhD"""
        self.__path2data = self.path2data
        folder_name = "04_Freiburg_Seidl2014"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, point)
        loaded_data = loadmat(self.path2file)

        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Penetrating"
        self.raw_data.data_lsb = 1 / loaded_data['GainPre'][0][0]
        self.raw_data.data_fs_orig = loaded_data['origFs'][0][0]

        self.raw_data.device_id = [0]
        elec_orig = np.arange(0, loaded_data['raw_data'].shape[0]).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for elec in elec_process:
            self.raw_data.data_raw.append(self.raw_data.data_lsb * np.float32(loaded_data['raw_data'][elec]))

        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = loaded_data['raw_data'].shape[1] / self.raw_data.data_fs_orig

        # Groundtruth
        self.raw_data.label_exist = False
        self.raw_data.spike_offset_us = [0]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_marre2018(self, case: int, point: int) -> None:
        # Link to data: https://zenodo.org/record/1205233#.YrBYrOzP1PZ
        self.__path2data = self.path2data
        folder_name = "05_Zenodo_Marre2018"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, point)
        loaded_data = loadmat(self.path2file)

        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Intracellular Matching"
        self.raw_data.data_lsb = float(loaded_data['Gain'][0])
        self.raw_data.data_fs_orig = int(loaded_data['fs'][0])

        self.raw_data.device_id = [0]
        # Information for data structure(Channel No = 255: Müll, No = 254: Intracellular response)
        elec_orig = np.arange(0, loaded_data['juxta_channel'][0]).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for elec in elec_process:
            self.raw_data.data_raw.append(self.raw_data.data_lsb * (np.float32(loaded_data['data'][:, elec]) - 32767))

        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = loaded_data['data'].shape[0] / self.raw_data.data_fs_orig

        # Groundtruth
        # TODO: Verarbeitung der Intrazellulären Antwort des juxta_channels
        self.raw_data.label_exist = False
        self.raw_data.spike_offset_us = [0]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_klaes_utah_array(self, case: int, nsp_device: int) -> None:
        """Loading the merged data file (from *.ns6 and *.nev files) from recordings with Utah electrode array
        from Blackrock Neurotechnology (case = experiment, nsp_device)"""
        self.__path2data = self.path2data
        folder_name = "06_Klaes_Caltech"
        data_type = '*_MERGED.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, nsp_device)
        loaded_data = loadmat(self.path2file, mat_dtype=True)

        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Utah"
        gain_str = loaded_data['rawdata']['LSB'][0, 0][0][0:-1].split(" ")
        if not len(gain_str) == 1:
            if gain_str[1] == 'm':
                gain_base = 1e-3
            elif gain_str[1] == 'u':
                gain_base = 1e-6
            elif gain_str[1] == 'n':
                gain_base = 1e-9
            else:
                gain_base = 1e0
        else:
            gain_base = 1e0
        self.raw_data.data_lsb = gain_base * float(gain_str[0])
        self.raw_data.data_fs_orig = int(loaded_data['rawdata']['SamplingRate'][0, 0][0])

        self.raw_data.device_id = [nsp_device]
        elec_orig = np.arange(0, int(loaded_data['rawdata']['NoElectrodes'][0, 0][0])).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        # TODO: Daten vom Utah-Array einlesen (Zwei Devices)
        data_raw = np.transpose(loaded_data['rawdata']['spike'][0, 0])
        for elec in elec_process:
            self.raw_data.data_raw.append(self.raw_data.data_lsb * np.float32(data_raw[elec]))
        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = data_raw.shape[0]

        # --- Groundtruth from BlackRock
        self.raw_data.label_exist = int(loaded_data['nev_detected']['Exits'][0, 0][0])
        self.raw_data.spike_offset_us = [0]
        self.nev_waveform = list()
        for elec in elec_process:
            self.raw_data.spike_offset_us.append(100)
            self.raw_data.spike_xpos.append(loaded_data['nev_detected'][f'Elec{1+elec}'][0, 0]['timestamps'][0, 0][0, :])
            self.raw_data.cluster_id.append(loaded_data['nev_detected'][f'Elec{1+elec}'][0, 0]['cluster'][0, 0][0, :])
            self.nev_waveform.append(self.raw_data.data_lsb * loaded_data['nev_detected'][f'Elec{1+elec}'][0, 0]['waveform'][0, 0])

        # --- Behaviour
        # TODO: Daten vom Utah-Array einlesen (Verhaltensanalyse)
        self.raw_data.behaviour_exist = True
        self.raw_data.behaviour = loaded_data['behaviour']
        del loaded_data

    def __load_rgc_tdb(self, case: int, point: int) -> None:
        """Loading the transient files from the Retinal Ganglian Cell Transient Database (RGC TDB)"""
        self.__path2data = self.path2data
        folder_name = "07_RGC_TDB"
        data_type = '*.mat'
        self.__prepare_access_subfolder(folder_name, data_type, 0, point)
        loaded_data = loadmat_mat73(self.path2file)

        # Pre-Processing: Remove empty entries and runs with only one spike
        check_xpos = loaded_data['sp_trains']['sp']
        check_data = loaded_data['sp_trains']['data']
        used_ch = list()
        for idx, pos in enumerate(check_xpos):
            if not isinstance(pos[0], str) and pos[0] is not None and check_data[idx][0] is not None:
                if pos[0].ndim == 1:
                    used_ch.append(idx)
        del check_data, check_xpos

        # Pre-Processing: Getting only the desired channels
        elec_orig = used_ch
        if not len(self.select_electrodes) == 0:
            elec_process = list()
            for elec in self.select_electrodes:
                elec_process.append(elec_orig[elec])
        else:
            elec_process = elec_orig

        spike_xpos = list()
        data_raw = list()
        for elec in elec_process:
            spike_xpos.append(loaded_data['sp_trains']['sp'][elec][0].astype('int'))
            data_raw.append(loaded_data['sp_trains']['data'][elec][0])

        # Input and meta --- This type are no electrode simultanously. It is more the experiment run
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Isolated RGC"
        self.raw_data.data_lsb = 1e-6
        self.raw_data.data_fs_orig = int(loaded_data['sp_trains']['sample_rate'][0][0])

        self.raw_data.device_id = [0]
        self.raw_data.electrode_id = np.arange(0, len(elec_process)).tolist()
        for idx, pos_ch in enumerate(elec_process):
            self.raw_data.data_raw.append(self.raw_data.data_lsb * np.float32(data_raw[idx]-data_raw[idx][0]))
        self.raw_data.data_time = self.raw_data.data_raw[0].shape[0] / self.raw_data.data_fs_orig

        # Groundtruth
        rgc_translator = RGC_Cell_Names()
        self.raw_data.label_exist = True
        self.raw_data.spike_offset_us = [-500]
        for idx, pos_ch in enumerate(elec_process):
            self.raw_data.spike_xpos.append(spike_xpos[idx])
            num_spikes = len(spike_xpos[idx])
            id = rgc_translator.get_id_from_cell_type(loaded_data['sp_trains']['cell_type'][pos_ch][0])
            self.raw_data.cluster_id.append(np.zeros(shape=(num_spikes, ), dtype=int) + id)

        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_fzj_mcs(self, case: int, point: int) -> None:
        """Loading the recording files from MCS setup in FZ Juelich (case = experiment, point = file)"""
        self.__path2data = self.path2data
        folder_name = "08_RGC_FZJuelich"
        data_type = '*_merged.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, point)
        loaded_data = loadmat_mat73(self.path2file)

        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "MCS 60MEA"
        self.raw_data.data_lsb = float(loaded_data['gain'])
        self.raw_data.data_fs_orig = float(loaded_data['fs'])

        self.raw_data.device_id = [0]
        elec_orig = np.arange(0, loaded_data['electrode'].shape[1]).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for elec in elec_process:
            self.raw_data.data_raw.append(self.raw_data.data_lsb * np.float32(loaded_data['electrode'][:, elec]))
        self.raw_data.data_time = loaded_data['electrode'].shape[0] / self.raw_data.data_fs_orig
        self.raw_data.electrode_id = elec_process
        # Groundtruth
        self.raw_data.label_exist = False
        self.raw_data.spike_offset_us = [0]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_musall_neuropixel(self, case: int, point: int) -> None:
        """Loading the files from recordings with NeuroPixel probes"""
        self.__path2data = self.path2data
        folder_name = "07_RGC_TDB"
        data_type = '*.mat'
        self.__prepare_access_subfolder(folder_name, data_type, case, point)
        loaded_data = loadmat_mat73(self.path2file)
        # TODO: Auswertung von NeuroPixel probes einlesen
        del loaded_data

        print("NOT IMPLEMENTED")
