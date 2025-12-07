import sys
import numpy as np
from scipy.io import loadmat

class DataHandler:
    """Class with data and meta information of the used neural dataset"""
    # --- Meta Information
    data_fs_orig = 0 # Original Sampling Frequency
    data_fs_current = 0  # Current sampling frequency of the data 
    data_lsb = 1.0 # Factor to convert raw values to Volt
    data_total_time_orig = 0.0 # Total time of recording in seconds
    device_id = str()  # Num of devices
    electrode_id = list() # Num of electrodes per device

    data_raw = list() # Data 

    # --- GroundTruth: Spike Sorting (per Channel)
    label_exist = False 
    spike_offset_us = list()
    spike_ovrlap = list()
    spike_xpos = list() # Position of spikes in samples 
    cluster_id = list() # Information about the identity of the spikes (neuron 1, neuron 2)
    # --- Behaviour (in total of MEA)
    behaviour_exist = False
    behaviour = None

    def execute_data_call(self):
        """Loading the dataset"""
        if "Martinez2009" in self.path:
            self.__load_martinez2009()
        elif "Pedreira2012" in self.path:
            self.__load_pedreira2012(0, 0)
        elif "Quiroga2020" in self.path:
            self.__load_quiroga2020(0, 0)
        elif "SineWave_Test" in self.path:
            self._load_sinusWave()
        else:
             self.logger.CRITICAL(f"Data type in path '{self.path}' is not recognized.")
             sys.exit()

    def __load_martinez2009(self) -> None:
        """Loading synthethic files from Quiroga simulation (2009)"""
        loaded_data = loadmat(self.path) #Read data from .mat file
        # Input and meta
        self.data_lsb = 0.5e-6 
        self.data_fs_orig = self.data_fs_current = int(1 / loaded_data["samplingInterval"][0][0] * 1000) #Sampling Frequency in Hz

        self.device_id = [0]
        self.electrode_id = [int(loaded_data["chan"][0])-1]
        self.data_raw = [self.data_lsb * np.float32(loaded_data["data"][0])]
        self.data_total_time_orig = loaded_data["data"][0].size / self.data_fs_orig
        # Groundtruth
        self.label_exist = True
        self.spike_offset_us = [-100]
        self.spike_ovrlap = list()
        self.spike_xpos = [(loaded_data["spike_times"][0][0][0])]
        self.cluster_id = [(loaded_data["spike_class"][0][0][0])]
        # Behaviour
        self.behaviour_exist = False
        self.behaviour = None
        del loaded_data

    def _load_sinusWave(self):
        """Loading synthethic sinus wave data for testing purposes"""
        sine_amplitude = 1
        sine_frequency = 50
        sine_phaseshift = 0 # Offset from zero
        sine_fs_sim = 1000  # Sampling frequency in Hz
        sine_t_sim = 2  # Total time in seconds

        total_pointsX = int(sine_fs_sim * sine_t_sim)
        time_points = np.linspace(0, sine_t_sim, total_pointsX, endpoint=False) #with endpoint False the last point is excluded; all points have same distance to each other
        sine_wave = sine_amplitude * np.sin(2 * np.pi * sine_frequency * time_points + sine_phaseshift)

        # Input and meta
        self.data_fs_orig = self.data_fs_current = sine_fs_sim # Original and current Sampling Frequency
        self.data_lsb = 1.0
        self.data_total_time_orig = sine_t_sim
        self.device_id = [0]
        self.electrode_id = [0]
        self.data_raw = [np.array(sine_wave, dtype=np.float32)]
        self.label_exist = False