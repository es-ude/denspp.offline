import os, sys
import glob
import dataclasses
import numpy as np
from typing import Tuple
from scipy.io import loadmat
from fractions import Fraction
from scipy.signal import resample_poly

@dataclasses.dataclass
class SettingsDATA:
    """Class for configuring the dataloader
    input:
    path        - Path to data storage
    data_set    - Type of dataset
    data_point  - Number within the dataset
    t_range     - List of the given time range for cutting the data [x, y]
    ch_sel      - List of electrodes to use
    fs_resample - Resampling frequency of the datapoint
    """
    path: str
    data_set: int
    data_point: int
    # Angabe des zu betrachteten Zeitfensters [Start, Ende] in sec.
    t_range: list
    # Auswahl der Elektroden(= -1, ruft alle Daten auf)
    ch_sel: list
    fs_resample: float

class RecommendedSettingsDATA(SettingsDATA):
    """Recommended configuration for testing"""
    def __init__(self):
        super().__init__(
            path="C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten",
            data_set=1, data_point=0,
            t_range=0, ch_sel=-1,
            fs_resample=100e3
        )

# ----- Read Settings -----
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

# TODO: Meta-Information durch Ordner-Scan ermöglichen
class DataController:
    """Class for loading and manipulating the used dataset"""
    def __init__(self, setting: SettingsDATA) -> None:
        self.settings = setting
        self.raw_data = DataHandler()
        self.path2file = None

        # --- Meta-Information about datasets
        # Number of points in data_set
        self.dataset_numpoints = 0
        # if used_channel = -1 -> All data
        self.no_channel = 0
        self.__fill_factor = 1
        self.__scaling = 1

    def do_call(self):
        """Loading the dataset"""
        # --- Checking if path is available
        if os.path.exists(self.settings.path):
            print(f"... data path {self.settings.path} is available")
        else:
            print(f"... data path {self.settings.path} is not available! Please check")
            sys.exit()

        # ----- Read data input -----
        data_type = self.settings.data_set
        data_set = self.settings.data_point
        if data_type == 1:
            self.__load_Martinez2009(data_set)
        elif data_type == 2:
            self.__load_Pedreira2012(data_set)
        elif data_type == 3:
            self.__load_Quiroga2020(data_set)
        elif data_type == 4:
            self.__load_Seidl2012(data_set)
        elif data_type == 5:
            self.__load_FZJ_MCS(data_set)
        elif data_type == 6:
            self.__load_KlaesLab(data_set)

        self.no_channel = self.raw_data.noChannel
        self.raw_data.fs_used = self.raw_data.fs_orig
        self.__do_take_elec()
        print("... using data point:", self.path2file)

    def __do_take_elec(self):
        used_ch = self.settings.ch_sel
        if not used_ch == -1:
            sel_channel = used_ch
            self.raw_data.channel = used_ch

            rawdata = list()
            spike_xpos = list()
            spike_no = list()
            cluster_id = list()
            cluster_no = list()

            for idx in range(0, len(sel_channel)):
                rawdata.append(self.raw_data.raw_data[idx])
                spike_xpos.append(self.raw_data.spike_xpos[idx])
                spike_no.append(self.raw_data.spike_no[idx])
                cluster_id.append(self.raw_data.cluster_id[idx])
                cluster_no.append(self.raw_data.cluster_no[idx])

            self.raw_data.raw_data = rawdata
            self.raw_data.spike_xpos = spike_xpos
            self.raw_data.spike_no = spike_no
            self.raw_data.cluster_id = cluster_id
            self.raw_data.cluster_no = cluster_no
        else:
            self.raw_data.channel = range(0, self.no_channel)

    def do_cut(self):
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self.settings.t_range)

        rawdata = self.raw_data.raw_data
        spikepos_in = self.raw_data.spike_xpos
        cluster_in = self.raw_data.cluster_id

        rawdata_out = list()
        cluster_id_out = list()
        cluster_no_out = list()
        spike_xout = list()
        spike_nout = list()

        # --- Positionen ermitteln
        if t_range.size == 2:
            idx0 = int(t_range[0] * self.raw_data.fs_orig)
            idx1 = int(t_range[1] * self.raw_data.fs_orig)

            for idx, data_in in enumerate(rawdata):
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1])
                self.__fill_factor = (idx0 - idx1) / data_in.size

                # --- Cutting labeled informations
                if self.raw_data.label_exist:
                    # Find values from x-positions
                    idx2 = np.argwhere(spikepos_in >= idx0)
                    idx2 = 1 + int(idx2[0])
                    idx3 = np.argwhere(spikepos_in <= idx1)
                    idx3 = int(idx3[-1])
                else:
                    idx2 = 0
                    idx3 = -1

                    spike_xout.append(spikepos_in[idx][idx2:idx3])
                    spike_nout.append(spikepos_in[-1].size)
                    cluster_id_out.append(cluster_in[idx][idx2:idx3])
                    cluster_no_out.append(np.unique(cluster_in[-1]).size)

            # Übergabe
            self.raw_data.raw_data = rawdata_out
            self.raw_data.spike_xpos = spike_xout
            self.raw_data.spike_no = spike_nout
            self.raw_data.cluster_id = cluster_id_out
            self.raw_data.cluster_no = cluster_no_out

    def do_resample(self):
        """Do resampling all transient signals"""
        desired_fs = self.settings.fs_resample
        self.raw_data.fs_used = desired_fs
        do_resampling = bool(self.raw_data.fs_used != self.raw_data.fs_orig)

        data_out = list()
        spike_out = list()

        if do_resampling:
            u_safe = 5e-6
            (p, q) = self.__get_resample_ratio(self.raw_data.fs_orig, self.raw_data.fs_used)
            self.__scaling = p / q

            for idx, data_in in enumerate(self.raw_data.raw_data):
                spikepos_in = self.raw_data.spike_xpos[idx]

                # --- Resampling the input
                u_chck = np.mean(data_in[0:10])
                if np.abs((u_chck < u_safe) - 1) == 1:
                    du = u_chck
                else:
                    du = 0

                data_out.append(du + resample_poly(data_in - du, p, q))

                # --- Resampling the labeled information
                if self.raw_data.label_exist:
                    spike_out.append(np.array(self.__scaling * spikepos_in, dtype=int))

            self.raw_data.raw_data = data_out
            self.raw_data.spike_xpos = spike_out
        else:
            self.__scaling = 1

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        print(f"... using data set of: {self.raw_data.data_name}")
        print(
            f"... original sampling rate of {int(1e-3 * self.raw_data.fs_orig)} kHz (resampling to {int(1e-3 * self.raw_data.fs_used)} kHz)")
        print(
            f"... using {self.__fill_factor * 100:.2f}% of the data (time length of {self.raw_data.raw_data[-1].size / self.raw_data.fs_used / self.__fill_factor:.2f} s)")
        print(f"... data includes {self.raw_data.noChannel} number of electrode ({self.raw_data.data_type})")
        if self.raw_data.label_exist:
            print(f"... includes labels (noSpikes: {np.sum(self.raw_data.spike_no)} - noCluster: {self.raw_data.cluster_no[-1]})")
        else:
            print(f"... excludes labels")

    def get_data(self) -> DataHandler:
        return self.raw_data

    def __get_resample_ratio(self, fin: float, fout: float) -> Tuple[int, int]:
        calced_fs = fout / fin
        (p, q) = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()
        return (p, q)

    def __prepare_access(self, folder_name: str, data_type: str, sel_dataset: int) -> None:
        folder_content = glob.glob(os.path.join(self.settings.path, folder_name, data_type))
        folder_content.sort()
        self.dataset_numpoints = len(folder_content)
        try:
            file_data = folder_content[sel_dataset]
            self.path2file = os.path.join(self.settings.path, folder_name, file_data)
        except:
            print("--- Folder not available - Please check folder name! ---")

    def __prepare_access_klaes(self, folder_name: str, data_type: str, sel_dataset: int, sel_nsp: int) -> None:
        path2data = os.path.join(self.settings.path, folder_name)
        folder_content = glob.glob(os.path.join(self.settings.path, folder_name, data_type))
        folder_content.sort()
        folder_data = [name for name in os.listdir(path2data) if os.path.isdir(os.path.join(path2data, name))]
        file_data = folder_data[sel_dataset]

        path2data = os.path.join(path2data, file_data)
        self.__prepare_access(path2data, data_type, sel_nsp)

    def __load_Martinez2009(self, indices: int) -> None:
        folder_name = "01_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

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

    def __load_Pedreira2012(self, indices: int) -> None:
        folder_name = "02_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        prep_index = self.path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = os.path.join(self.settings.path, folder_name, "ground_truth.mat")

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

    def __load_Quiroga2020(self, indices: int) -> None:
        folder_name = "03_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self.__prepare_access(folder_name, data_type, indices)
        loaded_data = loadmat(self.path2file)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = int(1)
        data.gain = 100e-6 * 10 ** (0 / 20)
        data.fs_orig = float(1 / loaded_data["samplingInterval"][0][0] * 1000)
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
        data.spike_offset = [100]
        # Return
        self.raw_data = data

    def __load_Seidl2012(self, indices: int) -> None:
        folder_name = "04_Freiburg_Seidl2014"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, indices)
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

    def __load_FZJ_MCS(self, indices: int) -> None:
        folder_name = "05_FZJ_MCS"
        data_type = '*_new.mat'
        self.__prepare_access(folder_name, data_type, indices)
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

    def __load_KlaesLab(self, indices: int) -> None:
        folder_name = "10_Klaes_Caltech"
        data_type = '*_MERGED.mat'
        self.__prepare_access_klaes(folder_name, data_type, indices, 0)
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
        for idx in range(1, data.noChannel+1):
            str_out = 'Elec'+ str(idx)
            A = (loaded_data['nev_detected'][str_out][0, 0]['timestamps'][0, 0][0, :])
            B = (loaded_data['nev_detected'][str_out][0, 0]['cluster'][0, 0][0, :])
            C = len(A)
            data.spike_xpos.append(A)
            data.spike_no.append(C)
            data.cluster_id.append(B)
            data.cluster_no.append(np.unique(B).size)
            data.spike_offset.append(100)
        # Return
        self.raw_data = data
