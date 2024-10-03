import sys
from os.path import join, exists
from glob import glob
import dataclasses
import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
import os
import csv
import matplotlib.pyplot as plt
import mplcursors
import pickle

from package.structure_builder import create_folder_general_firstrun


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


class _DataController:
    """Class for loading and manipulating the used dataset"""
    raw_data: None
    settings: SettingsDATA
    path2file: str

    def __init__(self) -> None:
        create_folder_general_firstrun()
        # --- Meta-Information about datasets
        # Information of subfolders and files
        self._no_subfolder = 0
        self._no_files = 0
        self.__fill_factor = 1
        self.__scaling = 1

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self.settings.t_range)
        rawdata = self.raw_data.data_raw
        spikepos_in = self.raw_data.evnt_xpos
        cluster_in = self.raw_data.evnt_cluster_id

        rawdata_out = list()
        spike_cout = list()
        spike_xout = list()

        if self.raw_data.data_fs_used == 0:
            self.raw_data.data_fs_used = self.raw_data.data_fs_orig

        # --- Positionen ermitteln
        if t_range.size == 2:
            idx0 = int(t_range[0] * self.raw_data.data_fs_used)
            idx1 = int(t_range[1] * self.raw_data.data_fs_used)
            self.__fill_factor = (idx0 - idx1) / rawdata[-1].size

            for idx, data_in in enumerate(rawdata):
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1])

                # --- Cutting labeled information
                if self.raw_data.label_exist:
                    # Adapting new data
                    idx2 = int(np.argwhere(spikepos_in[idx] >= idx0)[0])
                    idx3 = int(np.argwhere(spikepos_in[idx] <= idx1)[-1])
                    spike_xout.append(spikepos_in[idx][idx2:idx3] - idx0)
                    spike_cout.append(cluster_in[idx][idx2:idx3])

            # --- Return adapted data
            self.raw_data.data_raw = rawdata_out
            self.raw_data.evnt_xpos = spike_xout
            self.raw_data.evnt_cluster_id = spike_cout

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
                    spikepos_in = self.raw_data.evnt_xpos[idx]
                    spike_out.append(np.array(self.__scaling * spikepos_in, dtype=int))

            self.raw_data.data_raw = data_out
            self.raw_data.evnt_xpos = spike_out
        else:
            self.raw_data.data_fs_used = self.raw_data.data_fs_orig
            self.__scaling = 1

    def do_mapping(self) -> None:
        #TODO: Set Path from external
        csv_folder = r'C:/Users/Leoni Kaiser/Documents/Studium/Master/3. Semester/CPS_Projekt/Elektrodenmapping'
        if self.raw_data.data_type == "MCS 60MEA":
            print("Electrode geometry MCS 60MEA will be loaded.")
            csv_filename = 'MCS_60MEA.csv'
        else:
            csv_filename = '*.csv'
        mea = self._transform_rawdata_mapping(csv_folder, csv_filename)
        self._plot_data(mea)

    def output_meta(self) -> None:
        """Print some meta information into the console"""
        print(f"... using data set of: {self.raw_data.data_name}"
              "\n... using data point:", self.path2file)
        if not self.raw_data.data_fs_used == 0 and not self.raw_data.data_fs_used == self.raw_data.data_fs_orig:
            fs_addon = f" (resampling to {int(1e-3 * self.raw_data.data_fs_used)} kHz)"
        else:
            fs_addon = ""
        print(f"... original sampling rate of {int(1e-3 * self.raw_data.data_fs_orig)} kHz{fs_addon}"
              f"\n... using {self.__fill_factor * 100:.2f}% of the data "
              f"(time length of {self.raw_data.data_time / self.__fill_factor:.2f} s)")

        if self.raw_data.label_exist:
            cluster_array = None
            # Extract number of cluster size in all inputs
            for idx, clid in enumerate(self.raw_data.evnt_cluster_id):
                if idx == 0:
                    cluster_array = clid
                else:
                    cluster_array = np.append(cluster_array, clid)
            cluster_no = np.unique(cluster_array)

            # Extract number of spikes in all inputs
            num_spikes = 0
            for idx, spk_num in enumerate(self.raw_data.evnt_xpos):
                num_spikes += spk_num.size

            print(f"... includes labels (noSpikes: {num_spikes} - noCluster: {cluster_no.size})")
        else:
            print(f"... has no labels / groundtruth")

    def get_data(self):
        """Calling the raw data with groundtruth of the called data"""
        return self.raw_data
        
    def generate_xpos_label(self, used_channel: int) -> np.ndarray:
        """Generating label ticks"""
        fs_used = self.raw_data.data_fs_used
        fs_orig = self.raw_data.data_fs_orig
        xpos_in = self.raw_data.evnt_xpos[used_channel]
        return xpos_in / fs_orig * fs_used
        
    def generate_label_stream_channel(self, used_channel: int, window_time=1.6e-3) -> np.ndarray:
        """"""
        window_size = int(window_time * self.raw_data.data_fs_used)
        trgg0 = np.zeros(self.raw_data.data_raw[used_channel], dtype=int)
        for val in self.generate_xpos_label(used_channel):
            trgg0[int(val):int(val) + window_size] = 1
        return trgg0

    def generate_label_stream(self, window_time=1.6e-3) -> list:
        """"""
        trgg_out = list()
        for ch_used, trgg_used in enumerate(self.raw_data.evnt_xpos):
            trgg_out.append(self.generate_label_stream_channel(ch_used, window_time))
        return trgg_out

    def _prepare_call(self) -> None:
        """Loading the dataset"""
        # --- Checking if path is available
        if not exists(self.settings.path):
            print(f"... data path {self.settings.path} is not available! Please check")
            sys.exit()

    def _prepare_access_file(self, folder_name: str, data_type: str) -> None:
        """Getting the file of the corresponding trial"""
        sel_datacase = self.settings.data_case
        sel_datapoint = self.settings.data_point

        # --- Finding the right folder in data storage
        folder_structure = glob(join(self.settings.path, '*'))
        path2folder = ""
        for folder in folder_structure:
            if folder_name in folder:
                path2folder = folder

        # --- Checking for subfolder and file
        if path2folder:
            folder_structure = glob(join(path2folder, '*'))
            folder_content = glob(join(path2folder, data_type))

            if len(folder_content) == 0:
                # --- Taking datacase into account
                folder_content = glob(join(path2folder, folder_structure[sel_datacase], data_type))
                folder_content.sort()

            # --- Taking file
            try:
                self.path2file = folder_content[sel_datapoint]
                self._no_files = len(folder_content)
            except:
                print("--- Files are not available - Please check folder name! ---")
        else:
            print("--- Folder not available - Please check folder name! ---")

    def _read_csv_file(self, path2csv: str, num_channels: int, split_option: str) -> list:
        """"""
        loaded_data = [[] for idx in range(num_channels)]
        file = open(path2csv, 'r')

        for line in file:
            input = line.split(split_option)
            sel_list = 0
            for val in input:
                if val:
                    loaded_data[sel_list].append(val)
                    sel_list += 1
        return loaded_data

    def _transform_rawdata_from_csv_to_numpy(self, data: list) -> np.ndarray:
        """"""
        # --- Getting meta information
        num_channels = len(data) + 1

        num_samples = list()
        for idx, data0 in enumerate(data):
            num_samples.append(len(data0))
        num_samples = np.array(num_samples)
        num_channels = idx + 1

        # --- Getting data in right format
        data_used = np.zeros((num_channels, num_samples.min()), dtype=float)
        for idx, data_ch in enumerate(data):
            data_ch0 = list()
            for value in data_ch:
                data_ch0.append(float(value))
            data_used[idx, :] = np.array(data_ch0[0:num_samples.min()])

        return data_used

    def _transform_rawdata_to_numpy(self) -> None:
        """Transforming the initial raw data from list to numpy array"""
        if isinstance(self.raw_data.data_raw, list):
            num_channels = len(self.raw_data.data_raw)
            num_samples = np.zeros((num_channels, ), dtype=int)
            for idx, data in enumerate(self.raw_data.data_raw):
                num_samples[idx] = data.shape[0]

            data_out = np.zeros((num_channels, num_samples.min()), dtype=float)
            for idx, data in enumerate(self.raw_data.data_raw):
                data_out[idx, :] = data[0:num_samples.min()]

            self.raw_data.data_raw = data_out
        else:
            print("\t transformation may be already done - Please check!")

    def _read_csv_file_mapping(self, csv_path):
        numbers_list = []

        # Check if the CSV file exists
        if os.path.exists(csv_path):
            # Open the CSV file
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)

                # Iterate over each row in the CSV file
                for row in reader:
                    # Convert each element in the row to an integer (assuming numbers are expected)
                    row = row[0]
                    sep_row = row.split(';')
                    numbers_row = [int(value) for value in sep_row]
                    if numbers_row:
                        numbers_list.append(numbers_row)
        else:
            return
        return numbers_list

    def _transform_rawdata_mapping(self, csv_folder, csv_filename) -> np.array:
        """Transforming the numpy array input to 2D array with electrode mapping configuration"""
        data_in = self.raw_data.data_raw
        channel_head = self.raw_data.data_mapping
        csv_path = os.path.join(csv_folder, csv_filename)
        self.channel_numbers = self._read_csv_file_mapping(csv_path)
        if self.raw_data.data_type == "MCS 60MEA":
            mea = np.zeros((8, 8), dtype=object)
        else:
            print("Wrong data type. MEA won´t be initialized")
            return
        if isinstance(data_in, list):
            if (len(data_in) > 0) & (len(data_in) == len(channel_head)):
                print("2D->3D transformation will be done")
                # Map channel numbers
                for data_row_index, data_row in enumerate(self.channel_numbers):
                    # Iterate over each element in the data row
                    for data_col_index, data_value in enumerate(data_row):
                        # Get the preset number at the same position if available
                        if data_row_index < len(mea) and data_col_index < len(mea[data_row_index]):
                            (mea[data_row_index][data_col_index]) = data_value

                # Read channel data into 2D grid
                for x in range(0, mea.shape[0]):
                    for y in range(0, mea.shape[1]):
                        if mea[x, y] > 0:
                            column = 0
                            for channel in channel_head:
                                if int(channel[2:4]) == mea[x, y]:
                                    mea[x, y] = data_in[column]
                                    break
                                column += 1
                return mea
            else:
                print("Input data cannot be written into MEA")

    def _plot_data(self, mea):
        print("The plotted mea data is:", mea)
        num_rows, num_cols = mea.shape[0], mea.shape[1]
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8), gridspec_kw={'wspace': 0.8, 'hspace': 0.8})

        plotted_lines = []

        for i in range(num_rows):
            for j in range(num_cols):
                if isinstance(mea[i, j], np.ndarray):  # Check if element is an array
                    ax = axes[i, j]
                    line, = ax.plot(mea[i, j], 'b-')  # Plot the array as a line plot

                    # Set y-axis labels to the minimum and maximum values only
                    ymin, ymax = np.min(mea[i, j]), np.max(mea[i, j])
                    ax.set_yticks([ymin, ymax])
                    ymin = ymin * 10 ** 6
                    ymax = ymax * 10 ** 6
                    ax.set_yticklabels([f'{ymin:.2f}', f'{ymax:.2f}'])
                else:
                    ax = axes[i, j]
                    line, = ax.plot([0], 'b-')  # Plot the array as a line plot
                    ymin = 0
                    ymax = 0
                    ax.set_yticklabels([f'{ymin:.2f}', f'{ymax:.2f}'])
                    # Remove x-axis ticks and labels
                ax.set_xticklabels([])
                ax.set_xticks([])

                # Remove subplot border
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)

                if isinstance(mea[i, j], np.ndarray):
                    # Store the subplot and the data for the cursor event
                    line._mea_data = mea[i, j]
                else:
                    line._mea_data = [0]
                plotted_lines.append(line)

        plt.suptitle('MCS60 MEA Channel Signals [values upscaled with e5]', y=1)

        # Adjust layout and display the plot
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.tight_layout()

        # Function to create a bigger plot when clicking on a subplot
        def on_click(event):
            artist = event.artist
            data = artist._mea_data
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(data, 'bo-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Voltage')
            ax.set_title('Channel')
            plt.show()

        # Use mplcursors to enable click events
        cursor = mplcursors.cursor(plotted_lines, hover=False)
        cursor.connect("add", on_click)

        # Save the figure
        with open(
                r'C:/Users/Leoni Kaiser/Documents/Studium/Master/3. Semester/CPS_Projekt/Elektrodenmapping/Results/Plotted_Signals.fig.pickle',
                'wb') as f:
            pickle.dump(fig, f)

        plt.show()
        plt.close()

        for i in range(num_rows):
            for j in range(num_cols):
                if isinstance(mea[i, j], np.ndarray):
                    plt.plot(mea[i, j])
                    channel = self.channel_numbers[i][j]
                    channel = str(channel)
                    folder_save = r'C:/Users/Leoni Kaiser/Documents/Studium/Master/3. Semester/CPS_Projekt/Elektrodenmapping/Results/'
                    path_save = folder_save + 'channel_' + channel + '.png'
                    plt.savefig(path_save)
                    plt.close()


###########################################################################
if __name__ == "__main__":
    from package.data_call.call_spike_files import DataLoader, SettingsDATA

    settings = SettingsDATA(
        path="../../../2_Data",
        data_set=8, data_case=1, data_point=8,
        t_range=[0, 0.001], ch_sel=[], fs_resample=20e3
    )
    data_loader = DataLoader(settings)
    data_loader.do_call()
    data_loader.do_cut()
    data_loader.do_resample()
    data = data_loader.get_data()
    data_loader.do_mapping()
    del data_loader
    print(data)
