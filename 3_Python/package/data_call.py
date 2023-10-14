import os
import sys
import dataclasses
import numpy as np
from typing import Tuple
from fractions import Fraction
from scipy.signal import resample_poly
from package.data_load import DataLoader, DataHandler


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
    data_case: int
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
            path="D:\Data",
            data_set=1, data_case=0, data_point=0,
            t_range=0, ch_sel=-1,
            fs_resample=100e3
        )


# ----- Read Settings -----
class DataController(DataLoader):
    """Class for loading and manipulating the used dataset"""
    def __init__(self, setting: SettingsDATA) -> None:
        DataLoader.__init__(self)
        self.settings = setting
        self.path2data = setting.path
        self.path2file = str()
        self.raw_data = DataHandler()

        # --- Meta-Information about datasets
        # Information of subfolders and files
        self.no_subfolder = 0
        self.no_files = 0
        # if used_channel = -1 -> All data
        self.no_channel = 0
        self.__fill_factor = 1
        self.__scaling = 1

        # --- Waveform from NEV-File
        self.nev_waveform = []

    def do_call(self) -> None:
        """Loading the dataset"""
        # --- Checking if path is available
        if os.path.exists(self.settings.path):
            print(f"... data path {self.settings.path} is available")
        else:
            print(f"... data path {self.settings.path} is not available! Please check")
            sys.exit()

        # ----- Read data input -----
        data_type = self.settings.data_set
        data_set = self.settings.data_case
        data_point = self.settings.data_point

        self.execute_data_call(data_type, data_set, data_point)

        self.no_channel = self.raw_data.noChannel
        self.raw_data.fs_used = self.raw_data.fs_orig
        self.__do_take_elec()
        print("... using data point:", self.path2file)

    def __do_take_elec(self) -> None:
        used_ch = self.settings.ch_sel
        sel_channel = used_ch if not used_ch[0] == -1 else np.arange(0, self.no_channel)

        rawdata = list()
        spike_xpos = list()
        spike_no = list()
        cluster_id = list()
        cluster_no = list()

        for idx in sel_channel:
            rawdata.append(self.raw_data.raw_data[idx])

            if self.raw_data.label_exist:
                spike_xpos.append(self.raw_data.spike_xpos[idx])
                spike_no.append(self.raw_data.spike_no[idx])
                cluster_id.append(self.raw_data.cluster_id[idx])
                cluster_no.append(self.raw_data.cluster_no[idx])

        self.raw_data.raw_data = rawdata
        self.raw_data.spike_xpos = spike_xpos
        self.raw_data.spike_no = spike_no
        self.raw_data.cluster_id = cluster_id
        self.raw_data.cluster_no = cluster_no
        self.raw_data.channel = sel_channel

    def do_cut(self) -> None:
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
                    idx2 = int(np.argwhere(spikepos_in[idx] >= idx0)[0])
                    idx3 = int(np.argwhere(spikepos_in[idx] <= idx1)[-1])
                else:
                    idx2 = 0
                    idx3 = -1

                spike_xout.append(spikepos_in[idx][idx2:idx3])
                spike_nout.append(spikepos_in[idx].size)
                cluster_id_out.append(cluster_in[idx][idx2:idx3])
                cluster_no_out.append(np.unique(cluster_in[idx]).size)

            # Übergabe
            self.raw_data.raw_data = rawdata_out
            self.raw_data.spike_xpos = spike_xout
            self.raw_data.spike_no = spike_nout
            self.raw_data.cluster_id = cluster_id_out
            self.raw_data.cluster_no = cluster_no_out

    def do_resample(self) -> None:
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
                # --- Resampling the input
                u_chck = np.mean(data_in[0:10])
                if np.abs((u_chck < u_safe) - 1) == 1:
                    du = u_chck
                else:
                    du = 0

                data_out.append(du + resample_poly(data_in - du, p, q))

                # --- Resampling the labeled information
                if self.raw_data.label_exist:
                    spikepos_in = self.raw_data.spike_xpos[idx]
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
        """Calling the raw data with groundtruth of the called data"""
        return self.raw_data

    def __get_resample_ratio(self, fin: float, fout: float) -> Tuple[int, int]:
        calced_fs = fout / fin
        p, q = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()
        return p, q
