import sys
from os import listdir
from os.path import join, exists, isdir
from glob import glob
import dataclasses
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
from package.data.data_call_files import DataHandler, DataLoader


@dataclasses.dataclass
class SettingsDATA:
    """Class for configuring the dataloader
    input:
    path        - Path to data storage
    data_set    - Type of dataset
    data_point  - Number within the dataset
    t_range     - List of the given time range for cutting the data [x, y]
    ch_sel      - List of electrodes to use [empty=all]
    fs_resample - Resampling frequency of the datapoint
    """
    path: str
    data_set: int
    data_case: int
    data_point: int
    t_range: list
    ch_sel: list
    fs_resample: float


RecommendedSettingsDATA = SettingsDATA(
    path="../2_Data",
    data_set=1, data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
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
        # if self.select_electrode = -1 -> All data
        self.select_electrodes = setting.ch_sel
        self.no_channel = 0
        self.__fill_factor = 1
        self.__scaling = 1
        self.num_spikes = 0
        self.num_clusters = 0

        # --- Waveform from NEV-File
        self.nev_waveform = []

    def do_call(self) -> None:
        """Loading the dataset"""
        # --- Checking if path is available
        if not exists(self.settings.path):
            print(f"... data path {self.settings.path} is not available! Please check")
            sys.exit()

        # ----- Read data input -----
        data_type = self.settings.data_set
        data_set = self.settings.data_case
        data_point = self.settings.data_point

        self.execute_data_call(data_type, data_set, data_point)

        self.raw_data.data_fs_used = self.raw_data.data_fs_orig
        self.no_channel = len(self.raw_data.electrode_id)
        print("... using data point:", self.path2file)

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self.settings.t_range)

        rawdata = self.raw_data.data_raw
        spikepos_in = self.raw_data.spike_xpos
        cluster_in = self.raw_data.cluster_id

        rawdata_out = list()
        spike_cout = list()
        spike_xout = list()

        # --- Positionen ermitteln
        if t_range.size == 2:
            idx0 = int(t_range[0] * self.raw_data.data_fs_orig)
            idx1 = int(t_range[1] * self.raw_data.data_fs_orig)

            for idx, data_in in enumerate(rawdata):
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1])
                self.__fill_factor = (idx0 - idx1) / data_in.size

                # --- Cutting labeled information
                if self.raw_data.label_exist:
                    # Find values from x-positions
                    idx2 = int(np.argwhere(spikepos_in[idx] >= idx0)[0])
                    idx3 = int(np.argwhere(spikepos_in[idx] <= idx1)[-1])
                else:
                    idx2 = 0
                    idx3 = -1

                spike_xout.append(spikepos_in[idx][idx2:idx3])
                spike_cout.append(cluster_in[idx][idx2:idx3])

            # Übergabe
            self.raw_data.data_raw = rawdata_out
            self.raw_data.spike_xpos = spike_xout
            self.raw_data.cluster_id = spike_cout

    def do_resample(self) -> None:
        """Do resampling all transient signals"""
        desired_fs = self.settings.fs_resample
        do_resampling = bool(desired_fs != self.raw_data.data_fs_orig)

        data_out = list()
        spike_out = list()

        if do_resampling:
            self.raw_data.data_fs_used = desired_fs
            u_safe = 5e-6
            (p, q) = Fraction(self.raw_data.data_fs_used / self.raw_data.data_fs_orig).limit_denominator(10000).as_integer_ratio()
            self.__scaling = p / q

            for idx, data_in in enumerate(self.raw_data.data_raw):
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

            self.raw_data.data_raw = data_out
            self.raw_data.spike_xpos = spike_out
        else:
            self.__scaling = 1

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        print(f"... using data set of: {self.raw_data.data_name}")
        print(f"... original sampling rate of {int(1e-3 * self.raw_data.data_fs_orig)} kHz "
              f"(resampling to {int(1e-3 * self.raw_data.data_fs_used)} kHz)")
        print(f"... using {self.__fill_factor * 100:.2f}% of the data "
              f"(time length of {self.raw_data.data_time / self.__fill_factor:.2f} s)")
        print(f"... data includes {self.no_channel} number of electrode ({self.raw_data.data_type})")

        if self.raw_data.label_exist:
            cluster_array = None
            # Extract number of cluster size in all inputs
            for idx, clid in enumerate(self.raw_data.cluster_id):
                if idx == 0:
                    cluster_array = clid
                else:
                    cluster_array = np.append(cluster_array, clid)
            cluster_no = np.unique(cluster_array)
            # Extract number of spikes in all inputs
            for idx, spk_num in enumerate(self.raw_data.spike_xpos):
                self.num_spikes += spk_num.size

            print(f"... includes labels (noSpikes: {self.num_spikes} - noCluster: {cluster_no.size})")
        else:
            print(f"... excludes labels")

    def get_data(self) -> DataHandler:
        """Calling the raw data with groundtruth of the called data"""
        return self.raw_data

    def _prepare_access_file(self, folder_name: str, data_type: str, sel_datapoint: int) -> None:
        """Getting the file of the corresponding trial"""
        path = join(self.path2data, folder_name, data_type)
        folder_content = glob(path)
        folder_content.sort()
        self.no_files = len(folder_content)
        try:
            self.path2file = folder_content[sel_datapoint]
        except:
            print("--- Folder not available - Please check folder name! ---")

    def _prepare_access_folder(self, folder_name: str, data_type: str,
                                   sel_dataset: int, sel_datapoint: int) -> None:
        """Getting the file structure within cases/experiments in one data set"""
        path2data = join(self.path2data, folder_name)
        folder_data = [name for name in listdir(path2data) if isdir(join(path2data, name))]
        folder_data.sort()
        file_data = folder_data[sel_dataset]

        path2data = join(path2data, file_data)
        self._prepare_access_file(path2data, data_type, sel_datapoint)
        self.no_subfolder = len(file_data)
