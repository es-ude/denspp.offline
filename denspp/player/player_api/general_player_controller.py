import sys
import yaml
import logging
import inspect
from hardware_settings import *
import output_devices
from data_call_common import DataController
import debug_help_functions as dhf
from denspp.offline.data_call.call_handler import ControllerData, DataHandler, SettingsData
from call_handler_player import PlayerControllerData

default_config_path_to_yaml = "/Users/nickskill/Documents/Masterarbeit_Git/denspp.offline/denspp/player/hardware_config.yaml"

class GeneralPlayerController:
    logger: object # Logger object for logging messages


    logging_lvl: str # user defined logging level
    data_path: str # data input path
    hardware_config_values: dict # hardware configuration values
    data_config_values: dict # data configuration values
    data_config_do_cut: bool # whether to perform data cutting
    data_config_do_resample: bool # whether to perform data resampling

    _deployed_settingsData: SettingsData # deployed SettingsData object, holding data loading settings
    _deployed_playerControllerData: PlayerControllerData # deployed PlayerControllerData object, holding the _deployed_settingsData object


    hardware_controller: Hardware_settings # hardware controller settings (own class)
    deployed_data_controller: DataController # data controller for loading and manipulating dataset
    deployed_board_dataset: Board_dataset # board dataset for outputting data to hardware

    def __init__(self):
        self.logger =self._init_logging()
        self.config_path = self._read_sys_args()
        self.general_config = self._open_config_file()
        self._read_config()
        self._user_set_logging_level()

        self._config_hardware_controller()
        self._deployed_settingsData = self._config_call_handler_SettingsData()
        self._deployed_playerControllerData = self._config_call_handler_ControllerData()
        
        self.cut_data()
        self.resample_data()

        self._deployed_board_dataset = self._config_board_dataset()
        self._load_board_dataset_into_hardware_settings()
        self.transfer_data_to_vertical_resolution()
        self.output_data_for_hardware()
    

    def _init_logging(self) -> None:
        """Initialize logging configuration"""
        log_format = '%(asctime)s - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(__name__)
        return logger
    

    def _user_set_logging_level(self) -> None:
        """Set logging level based on user configuration"""
        level = self.logging_lvl.upper()
        if level == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif level == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif level == "ERROR":
            self.logger.setLevel(logging.ERROR)
        elif level == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        else:
            self.logger.warning(f"Unknown logging level: {level}. Defaulting to INFO.")
            self.logger.setLevel(logging.INFO)


    def _read_sys_args(self) -> str:
        global default_config_path_to_yaml
        """Read system arguments for configuration path"""
        if len(sys.argv) > 1:
            tempPath = sys.argv[1]
        else:
            tempPath = default_config_path_to_yaml
        self.logger.info(f"The following config Path is used: {tempPath}")
        return tempPath


    def _open_config_file(self) -> dict:
        """Open a YAML configuration file and load its contents."""
        try:
            with open(self.config_path , 'r') as stream:
                configLoaded = yaml.safe_load(stream)
        except FileNotFoundError:
            self.logger.critical(f"The File/Path '{self.config_path}' are not found")
            sys.exit()
        except yaml.YAMLError as e:
            self.logger.critical(f"Error parsing the YAML file: {e}")
            sys.exit()
        else:
            self.logger.info("Configuration file loaded successfully")
            return configLoaded


    def _read_config(self) -> None:
        """Read and process the general configuration settings."""
        self.logging_lvl = self.general_config["General_Configuration"]["Logging_Lvl"]
        self.data_path = self.general_config["Data_Input"]["input"]
        self.hardware_config_values = self.general_config["Hardware"]
        self.data_config_values = self.general_config["Data_Configuration"]
        self.data_config_do_cut = self.general_config["Data_Configuration"]["Data_Preprocessing"]["do_cut"]
        self.data_config_do_resample = self.general_config["Data_Configuration"]["Data_Preprocessing"]["do_resampling"]
        self.translation_value_voltage = self.general_config["Data_Configuration"]["Voltage_Scaling"]["translation_value_voltage"]
        self._hardware_data_channel_mapping = self.general_config["Data_Configuration"]["Hardware_Data_Mapping"]["channel_mapping"]


    def _config_hardware_controller(self) -> None:
        """Configure the hardware controller based on the selected device."""
        used_device = None
        
        for key, data in self.hardware_config_values.items(): # Check for multiple used devices, this is not allowed, and get the used device
            if data["used"]: 
                if used_device is not None:
                    self.logger.CRITICAL("Multiple hardware devices are set to be used. Please select only one device.")
                    sys.exit()
                used_device = (key, data)
        if used_device is None: # Check if a device is selected
            self.logger.CRITICAL("No hardware device is set to be used. Please select one device.")
            sys.exit()

        # Load the specific device settings
        used_device_class =None 
        for name, obj in inspect.getmembers(output_devices, inspect.isclass): #check if the specified device exists in output_devices.py
            if obj.__module__ == output_devices.__name__ and name == used_device[0]:
                used_device_class = obj
                break
        if used_device_class is None: # Device not found in output_devices.py
            self.logger.CRITICAL(f"The specified hardware device '{used_device}' is not defined in output_devices.py.")
            sys.exit()

        specific_class = getattr(output_devices, used_device[0]) # Get the class of the specific device
        specific_device_settings = specific_class() # Create an instance of the specific device class
        
        if hasattr(specific_device_settings, 'output_open'): # Set output_open if it exists in the device settings, needed for Oscilloscope
            specific_device_settings.output_open = used_device[1]["output_open"]

        self.hardware_controller = Hardware_settings(specific_device_settings, self.logger, self._hardware_data_channel_mapping) # Create hardware controller with specific device settings
        self.logger.info(f"Hardware controller configured for device: {used_device[0]}")


    def _config_call_handler_SettingsData(self) -> SettingsData:
        """Configure the SettingsData based on the current configuration, This Object is used to load data in the ControllData class

        Returns:
            SettingsData: Configured SettingsData object
        """        
        deployed_settingsData = SettingsData(pipeline ="PipelineV0",
                                                do_merge = False,
                                                path = "",
                                                data_set = self.data_path,
                                                data_case = 0,
                                                data_point = 0,
                                                ch_sel = self.data_config_values["Data_Preprocessing"]["channel_selection"],
                                                fs_resample = self.data_config_values["Data_Preprocessing"]["sampling_rate_resample"],
                                                t_range_sec = [self.data_config_values["Time"]["start_time"], self.data_config_values["Time"]["end_time"]] if self.data_config_values["Time"]["start_time"] is not None and self.data_config_values["Time"]["end_time"] is not None else [],
                                                do_mapping = False,
                                                is_mapping_str = False)
        return deployed_settingsData
        

    def _config_call_handler_ControllerData(self) -> PlayerControllerData:
        """Configure the PlayerControllerData based on the current configuration, with the SettingsData Object loaded

        Returns:
            PlayerControllerData: Configured PlayerControllerData object, with loaded data
        """        
        deployed_playerControllerData = PlayerControllerData()
        deployed_playerControllerData._settings = self._deployed_settingsData
        deployed_playerControllerData.do_call()
        return deployed_playerControllerData
    

    def cut_data(self) -> None:
        """Cut data using the DataController."""
        if self.data_config_do_cut:
            self._deployed_playerControllerData.do_cut()
            
            data = self._deployed_playerControllerData.get_data()
            self.logger.info(f"Plotted data with sampling rate {data.fs_used} with {len(data.data_raw)}")
        else:
            self.logger.info("Data cutting is disabled in the configuration.")


    def resample_data(self) -> None:
        """Resample data using the DataController."""
        print(self._deployed_playerControllerData._settings.fs_resample)
        print(self.hardware_controller._dac_max_sampling_rate)

        if self._deployed_playerControllerData._settings.fs_resample > self.hardware_controller._dac_max_sampling_rate:# Check if the desired sampling rate is supported by the hardware
            self.logger.CRITICAL(f"The desired resampling rate of {self._deployed_playerControllerData._settings.fs_resample} Hz exceeds the maximum supported rate of {self.hardware_controller._dac_max_sampling_rate} Hz for the selected hardware.")
            sys.exit()

        if self.data_config_do_resample:
            self._deployed_playerControllerData.do_resample()
            
            # Plot data after resampling, if the logging level is set to DEBUG
            if self.logging_lvl.upper() == "DEBUG":
                self.logger.debug(f"Plotted data with sampling rate {self._deployed_playerControllerData._raw_data.fs_used}")
                dhf.plot_data(self._deployed_playerControllerData._raw_data.data_raw[0], self._deployed_playerControllerData._raw_data.fs_used, self._deployed_playerControllerData._raw_data.time_end, "resampling")
            else:
                self.logger.info(f"Data resampling completed, new sampling rate: {self._deployed_playerControllerData._raw_data.fs_used} Hz")
        else:
            self.logger.info("Data resampling is disabled in the configuration.")

    def _config_board_dataset(self) -> Board_dataset:
        """Data that gone be output to the hardware device

        Returns:
            Board_dataset: Configured Board_dataset object
        """        
        data = self._deployed_playerControllerData.get_data()
        deployed_board_dataset = Board_dataset(_data= data.data_raw,
                                                    _samplingrate= data.fs_used,
                                                    _groundtruth= [] if data.label_exist else None,
                                                    _translation_value_voltage= self.translation_value_voltage)
        return deployed_board_dataset
    
    def _load_board_dataset_into_hardware_settings(self) -> None:
        """Load the board dataset into the hardware settings controller."""
        self.hardware_controller._data = self._deployed_board_dataset
    
    def transfer_data_to_vertical_resolution(self) -> None:
        """Transfer data to match the vertical resolution of the hardware."""

        if hasattr(self.hardware_controller, "_output_open"):
           transfer_data = self.hardware_controller.translate_data_for_oscilloscope()
        else:
            transfer_data = self.hardware_controller.translate_data_float2int(self.deployed_data_controller.data_raw)

        if self.logging_lvl.upper() == "DEBUG":
            data = self.hardware_controller.get_data
            dhf.plot_data(data._data[0],data._samplingrate, 1, "transferring")


    def output_data_for_hardware(self) -> None:
        """Output data to the configured hardware device."""
        if hasattr(self.hardware_controller, "_output_open"):
            self.logger.info("Outputting data for the Oscilloscope")
            self.hardware_controller.create_csv_for_MXO4()
        
        self.logger.info("Data output to hardware completed.")

if __name__ == "__main__":
    deployed_general_controller = GeneralPlayerController()