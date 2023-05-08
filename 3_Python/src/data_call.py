import os
import glob
import sys

import numpy as np
from typing import Tuple
from scipy.io import loadmat
from fractions import Fraction
from scipy.signal import resample_poly

# ----- Read Settings -----
class DataHandler:
    # --- Meta Information
    data_name = None
    data_type = None
    gain = None
    noChannel = None
    # --- Data
    fs_orig = None
    fs_used = None
    channel = list()
    raw_data = list()
    # --- Behaviour
    behaviour_exist = False
    behaviour = None
    # --- GroundTruth from other sources
    label_exist = False
    spike_xpos = list()
    spike_no = list()
    cluster_id = list()
    cluster_no = list()
    # --- GroundTruth from SpikeDeepClassifier
    sorted_exist = False

class DataController:
    def __init__(self, path2data: str, used_channel: int) -> None:
        # Meta-Information about datasets
        self.max_datapoints = np.array([5, 16, 22, 2, 1])
        # Settings
        self.raw_data = DataHandler()
        self.path2data = path2data
        self.path2file = None
        # if used_channel = 0 -> All data
        self.no_channel = 0
        self.used_channel = used_channel

        self.__fill_factor = 1
        self.__scaling = 1

    def do_call(self, data_type: int, data_set: int):
        # --- Checking if path is available
        if os.path.exists(self.path2data):
            print(f"... data path {self.path2data} is available")
        else:
            print(f"... data path {self.path2data} is not available! Please check")
            sys.exit()

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
            self.__load_FZJ_MCS(data_set)
        elif data_type == 6:
            self.__load_KlaesLab(data_set)

        self.no_channel = self.raw_data.noChannel
        self.raw_data.fs_used = self.raw_data.fs_orig
        self.__do_take_elec()
        print("... using data point:", self.path2file)

    def __do_take_elec(self):
        if not self.used_channel == -1:
            sel_channel = self.used_channel
            self.raw_data.channel = self.used_channel

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

    def do_cut(self, t_range: np.array):
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

    def do_resample(self, desired_fs: int):
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
                    spike_out = self.__scaling * spikepos_in

            self.raw_data.raw_data = data_out
            self.raw_data.spike_xpos = spike_out
        else:
            self.__scaling = 1

    def output_meta(self):
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

    def __get_resample_ratio(self, fin: int, fout: int) -> Tuple[int, int]:
        calced_fs = fout / fin
        (p, q) = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()
        return (p, q)

    def __prepare_access(self, folder_name: str, data_type: str, sel_dataset: int) -> None:
        folder_content = glob.glob(os.path.join(self.path2data, folder_name, data_type))
        folder_content.sort()
        try:
            file_data = folder_content[sel_dataset]
            self.path2file = os.path.join(self.path2data, folder_name, file_data)
        except:
            print("--- Folder not available - Please check folder name! ---")
    def __prepare_access_klaes(self, folder_name: str, data_type: str, sel_dataset: int, sel_nsp: int) -> None:
        path2data = os.path.join(self.path2data, folder_name)
        folder_content = glob.glob(os.path.join(self.path2data, folder_name, data_type))
        folder_content.sort()
        folder_data = [name for name in os.listdir(path2data) if os.path.isdir(os.path.join(path2data, name))]
        file_data = folder_data[sel_dataset]

        path2data = os.path.join(path2data, file_data)
        self.__prepare_access(path2data, data_type, sel_nsp)

    def __load_Martinez2009(self, indices: int = range(0, 4)) -> None:
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
        data.raw_data.append(data.gain * loaded_data["data"][0])
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = True
        data.spike_xpos.append(loaded_data["spike_times"][0][0][0])
        data.cluster_id.append(loaded_data["spike_class"][0][0][0])
        data.cluster_no.append(np.unique(data.cluster_id[0]).size)
        data.spike_no.append(data.spike_xpos[0].size)
        # Return
        self.raw_data = data

    def __load_Pedreira2012(self, indices: int = range(0, 15)) -> None:
        folder_name = "02_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self.__prepare_access(folder_name, data_type, indices)

        prep_index = self.path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = os.path.join(self.path2data, folder_name, "ground_truth.mat")

        loaded_data = loadmat(self.path2file)
        ground_truth = loadmat(path2label)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = "Synthetic"
        data.noChannel = int(1)
        data.gain = 25e-6 * 10 ** (0 / 20)
        data.fs_orig = int(24000)
        data.raw_data.append(data.gain * loaded_data["data"][0])
        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = True
        data.spike_xpos.append(ground_truth["spike_first_sample"][0][num_index - 1][0])
        data.cluster_id.append(ground_truth["spike_classes"][0][num_index - 1][0])
        data.cluster_no.append(np.unique(data.cluster_id[0]).size)
        data.spike_no.append(data.spike_xpos[0].size)
        # Return
        self.raw_data = data

    def __load_Quiroga2020(self, indices: int = range(0, 21)) -> None:
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
        data.fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        data.raw_data.append(data.gain * loaded_data["data"][0])
        # Behaviour
        data.behaviour_exist = False
        data.behaviour = None
        # Groundtruth
        data.label_exist = True
        data.spike_xpos.append(loaded_data["spike_times"][0][0][0])
        data.cluster_id.append(loaded_data["spike_class"][0][0][0]-1)
        data.cluster_no.append(np.unique(data.cluster_id[0]).size)
        data.spike_no.append(data.spike_xpos[0].size)
        # Return
        self.raw_data = data

    def __load_Seidl2012(self, indices: int = range(0, 1)) -> None:
        folder_name = "04_Freiburg_Seidl2014"
        data_type = '*.mat'
        self.__prepare_access(folder_name, data_type, indices)
        loaded_data = loadmat(self.path2file)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = loaded_data["TypeMEA"][0][0]
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
        # Return
        self.raw_data = data

    def __load_FZJ_MCS(self, indices: int = range(0, 1)) -> None:
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
            data.raw_data.append(raw[:,idx])

        # Behaviour
        data.behaviour_exist = False
        # Groundtruth
        data.label_exist = False
        # Return
        self.raw_data = data

    def __load_KlaesLab(self, indices: int = range(0, 22)) -> None:
        folder_name = "10_Klaes_Caltech"
        data_type = '*_MERGED.mat'
        self.__prepare_access_klaes(folder_name, data_type, indices, 0)
        loaded_data = loadmat(self.path2file, mat_dtype=True)

        data = DataHandler()
        # Input and meta
        data.data_name = folder_name
        data.data_type = loaded_data['rawdata']['Exits'][0, 0][0]
        data.noChannel = int(loaded_data['rawdata']['NoElectrodes'][0, 0][0])
        data.gain = 0.25e-6 #loaded_data['rawdata']['LSB'][0, 0][0]
        data.fs_orig = int(loaded_data['rawdata']['SamplingRate'][0, 0][0])

        raw = data.gain * loaded_data['rawdata']['spike'][0, 0]
        for idx in range(0, data.noChannel):
            data.raw_data.append(raw[:,idx])

        # --- Behaviour
        data.behaviour_exist = True
        data.behaviour = loaded_data['behaviour']
        # --- Groundtruth from BlackRock
        data.label_exist = int(loaded_data['nev_detected']['Exits'][0, 0][0])
        # Processing of electrode informations
        for idx in range(1, data.noChannel+1):
            str_out = 'Elec'+ str(idx)
            A = (loaded_data['nev_detected'][str_out][0, 0]['timestamps'][0, 0][0, :])
            B = (loaded_data['nev_detected'][str_out][0, 0]['cluster'][0, 0][0, :])
            C = len(A)
            data.spike_xpos.append(A)
            data.spike_no.append(C)
            data.cluster_id.append(B)
            data.cluster_no.append(np.unique(B).size)

        # --- Groundtruth from SpikeDeepClassifier
        data.sorted_exist = int(loaded_data['sorted']['Exits'][0, 0][0])
        # Return
        self.raw_data = data

    # TODO: Andere Quellen noch anpassen