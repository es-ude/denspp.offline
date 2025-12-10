import numpy as np
from fractions import Fraction
from scipy.signal import resample_poly
from data_call_files import DataHandler


# ----- Read Settings -----
class DataController(DataHandler):
    """Class for loading and manipulating the used dataset"""
    def __init__(self) -> None:
        self.logger :object #Logger object, defined in general_controller.py
        self.path :str #Path to dataset
        self.t_range :list #Time range for data selection
        self.ch_sel  :list #Channel selection for data processing
        self.fs_resample :int #Sampling rate for resampling

        # --- Meta-Information about datasets
        self.total_number_of_channels = 0
        self.__scaling = 1

        # --- Waveform from NEV-File
        self.nev_waveform = []

    def do_call(self) -> None:
        """Loading the dataset"""
        # ----- Read data input -----#
        self.execute_data_call()  #using the data_call_files.py for that
        self.total_number_of_channels = len(self.electrode_id)
        self.logger.info(f"Data loaded from: {self.path}")

    def do_cut(self) -> None:
        """Cutting all transient electrode signals in the given range"""
        t_range = np.array(self.t_range) # Convert to numpy array for easier handling

        # List to store the cut data
        rawdata_out = list()
        spike_cout = list()
        spike_xout = list()

        # --- Positionen ermitteln
        if t_range.size == 2: # Check whether a time range has been specified
            idx0 = int(t_range[0] * self.data_fs_current) # Start Value time multiplied with sampling rate
            idx1 = int(t_range[1] * self.data_fs_current) # End Value time multiplied with sampling rate

            for idx, data_in in enumerate(self.data_raw): #goes through all channels
                # --- Cutting specific time range out of raw data
                rawdata_out.append(data_in[idx0:idx1]) #slice the data according to the calculated indices

                # --- Cutting labeled information
                if self.label_exist:
                    # Find values from x-positions
                    idx2 = int(np.argwhere(self.spike_xpos[idx] >= idx0)[0])  #Find spike positions greater than start index
                    idx3 = int(np.argwhere(self.spike_xpos[idx] <= idx1)[-1]) #Find spike positions less than end index

                    spike_xout.append(self.spike_xpos[idx][idx2:idx3] - idx0) # with -idx0 the spike positions are adjusted to the new sliced data
                    spike_cout.append(self.cluster_id[idx][idx2:idx3]) # not with -idx0 because cluster IDs are independent of time

            self.data_raw = rawdata_out
            self.spike_xpos = spike_xout
            self.cluster_id = spike_cout

    def do_resample(self, u_safe = 5e-6, num_points_mean: int=100) -> None:
        """Do resampling all transient signals"""
        # u_safe: Threshold value for offset calculation (e.g., 5 µV)
        # num_points_mean: Number of points to calculate the mean value for offset correction
        
        data_out = list() #final resampled data
        spike_out = list() #final resampled spike positions

        if self.fs_resample != self.data_fs_current:
            # With is function a mathematical accurate ratio between the recoding and the desired sampling rate is calculated -> p/q (0.416666 = 5/12)
            (p, q) = Fraction(self.fs_resample / self.data_fs_current).limit_denominator(10000).as_integer_ratio()
            self.__scaling = p / q # Scaling factor for spike positions

            for idx, data_in in enumerate(self.data_raw): #goes through all channels
                # --- Resampling the input
                u_chck = np.mean(data_in[0:num_points_mean +1]) # Calculate the mean value for DC offset for every channel
                du = u_chck if np.abs(u_chck) > u_safe else 0.0

                #This function does the up- and downsampling in one step and includes a with u_check a Correction of a DC offset
                #The resampling function works better when there is no DC offset in the signal. 
                data_out.append(du + resample_poly(data_in - du, p, q, padtype='wrap')) 

                # --- Resampling the labeled information
                if self.label_exist:
                    spike_out.append(np.array(self.__scaling * self.spike_xpos[idx], dtype=int)) # These are the resampled spike positions

            self.data_raw = data_out # This are the resampled data 
            self.spike_xpos = spike_out # This are the resampled spike positions (These are positions where special events occur)
            self.data_fs_current = self.fs_resample #Setting the current sampling rate to the resampled one for the Dataset
        else:
            self.logger.warning("Resampling skipped because the desired sampling rate is equal to the current sampling rate.")
            self.__scaling = 1 # No resampling -> scaling = 1


# - - - - Graveyard - - - -
    def generate_xpos_label(self, used_channel: int) -> np.ndarray:
        """Generating label ticks"""
        fs_used = self.raw_data.data_fs_used
        fs_orig = self.raw_data.data_fs_orig
        
        dx_us = int(1e-6 * self.raw_data.spike_offset_us[0] * fs_used)
        xpos_in = self.raw_data.spike_xpos[used_channel]
        return (xpos_in/fs_orig * fs_used - dx_us)

    def generate_label_stream_channel(self, used_channel: int, window_time=1.6e-3) -> np.ndarray:
        """"""
        window_size = int(window_time * self.raw_data.data_fs_used)
        trgg0 = np.zeros(self.raw_data.data_raw[used_channel].size, dtype=int)
        for val in self.generate_xpos_label(used_channel):
            trgg0[int(val)+window_size:int(val) + 2*window_size] = 1
        return trgg0

    def generate_label_stream(self, window_time=1.6e-3) -> list:
        """"""
        trgg_out = list()
        for ch_used, trgg_used in enumerate(self.raw_data.spike_xpos):
            trgg_out.append(self.generate_label_stream_channel(ch_used, window_time))
        return trgg_out