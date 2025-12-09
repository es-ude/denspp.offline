from denspp.offline.data_call.call_handler import ControllerData, DataHandler, SettingsData
import numpy as np

class PlayerControllerData(ControllerData):
    """ControllerData specialized for the Player application."""

    def __init__(self):
        super().__init__()
        self._methods_available = dir(PlayerControllerData)


    def syntheticSineWave(self):
        """Loading synthethic sinus wave data for testing purposes"""
        sine_amplitude = 1
        sine_frequency = 50
        sine_phaseshift = 0 # Offset from zero
        sine_fs_sim = 1000  # Sampling frequency in Hz
        sine_t_sim = 2  # Total time in seconds

        total_pointsX = int(sine_fs_sim * sine_t_sim)
        time_points = np.linspace(0, sine_t_sim, total_pointsX, endpoint=False) #with endpoint False the last point is excluded; all points have same distance to each other
        sine_wave = sine_amplitude * np.sin(2 * np.pi * sine_frequency * time_points + sine_phaseshift)
        raw_data_array = np.array([sine_wave], dtype=np.float32)

        self._load_rawdata_into_pipeline(elec_type= "synthetic_sine",
                                         dataset_name= "SineWave_Test",
                                         file_name="SineWave_Test",
                                         fs_orig=sine_fs_sim,
                                         elec_orn = [0],
                                         rawdata= raw_data_array,
                                         scale_data= 1.0,
                                         evnt_pos= [],
                                         evnt_id= [])