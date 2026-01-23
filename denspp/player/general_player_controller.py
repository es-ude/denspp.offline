import sys, yaml, logging, copy
from pathlib import Path
from .data_translation import DataTranslator, BoardDataset
from .output_devices import hardware_specification_oscilloscope_mox4, hardware_specification_denspp_player, hardware_specification_denspp_player_sdcard, hardware_specification_denspp_player_import
from .debug_help_functions import plot_data
from denspp.offline.data_call.call_handler import SettingsData
from .call_handler_player import PlayerControllerData
from denspp.offline import get_path_to_project
from .signal_validation import SignalCompartor
from dataclasses import dataclass

@dataclass
class DensppGenerationPlayerConfig:
    """Configuration for the  General Player Controller

    Attributes:
        logging_lvl (str): Logging level for the controller
        input (Path): Path to the input data
        target_hardware (str): Target hardware device for playback
        output_open (bool): Whether the output is open (+/-10V) or 50 Ohm (+/-5V)
        start_time (float): Start time for data playback
        end_time (float): End time for data playback
        do_cut (bool): Whether to cut the data to the specified time range
        do_resample (bool): Whether to resample the data
        target_sampling_rate (int): Target sampling rate for resampling
        translation_value_voltage (float): Voltage translation value for the hardware
        channel_mapping (list): Channel mapping for the hardware device
    """    
    logging_lvl: str
    input: Path
    target_hardware: str
    output_open: bool
    start_time: float
    end_time: float
    do_cut: bool
    do_resample: bool
    target_sampling_rate: int
    translation_value_voltage: float
    channel_mapping: list


default_config_path_to_yaml = Path(get_path_to_project("config")) / "hardware_config.yaml"
class GeneralPlayerController:
    _logger: logging.Logger # Logger object for logging messages
    _config_path : Path # Path to configuration file

    _player_config: DensppGenerationPlayerConfig # Player configuration dataclass instance

    _deployed_settingsData: SettingsData # deployed SettingsData object, holding data loading settings
    _deployed_data_translator: DataTranslator # data translator settings (own class)
    _deployed_playerControllerData: PlayerControllerData # deployed PlayerControllerData object, holding the _deployed_settingsData object
    _untreated_raw_data: BoardDataset # untreated raw data loaded from the PlayerControllerData object

    _deployed_board_dataset: BoardDataset # board dataset for outputting data to hardware


    def __init__(self, class_config: DensppGenerationPlayerConfig = None):
        if class_config:
            """Init process for the configuration class"""
            self._player_config = class_config
            self._logger =self._init_logging()
            self._user_set_logging_level()

            self._deployed_settingsData = self._config_call_handler_SettingsData()
            self._deployed_data_translator = self._config_data_translator()
            self._deployed_playerControllerData = self._config_call_handler_ControllerData()
            self._untreated_raw_data = copy.deepcopy(self._deployed_playerControllerData.get_data())

            self._cut_data()
            self._untreated_raw_data_with_cut = copy.deepcopy(self._deployed_playerControllerData.get_data())
            self._resample_data()
            self._create_and_load_board_dataset_into_translator()
            self._produce_data_for_hardware()

        else:
            """Init porcess for the yaml configuration file"""
            self._logger =self._init_logging()
            self._config_path = self._read_sys_args_and_set_path()
            
            self._player_config = self._read_config()
            
            self._user_set_logging_level()

            self._deployed_settingsData = self._config_call_handler_SettingsData()
            self._deployed_data_translator = self._config_data_translator()
            self._deployed_playerControllerData = self._config_call_handler_ControllerData()
            self._untreated_raw_data = copy.deepcopy(self._deployed_playerControllerData.get_data())

            self._cut_data()
            self._untreated_raw_data_with_cut = copy.deepcopy(self._deployed_playerControllerData.get_data())
            self._resample_data()

            self._create_and_load_board_dataset_into_translator()
            self._produce_data_for_hardware()


    @property
    def get_untreated_data(self) -> BoardDataset:
        """Output untreated raw data in BoardDataset format

        Returns:
            BoardDataset: Holdes untreated raw data, data name, data type, electrode id, fs original and fs used
        """        
        return self._untreated_raw_data


    @property
    def get_untreated_data_with_cut(self) -> BoardDataset:
        """Output untreated raw data with cut in BoardDataset format

        Returns:
            BoardDataset: Holdes untreated raw data with cut, data name, data type, electrode id, fs original and fs used
        """        
        return self._untreated_raw_data_with_cut


    @property
    def get_processed_data(self) -> BoardDataset:
        """Output processed data in BoardDataset format

        Returns:
            BoardDataset: Holdes processed data and some metadata
        """        
        return self._deployed_data_translator.get_data

    
    def _init_logging(self) -> logging.Logger:
        """Initialize logger, this object is used for logging messages throughout the application

        Returns:
            logging.Logger: Object for logging messages
        """
        log_format = '%(asctime)s - %(levelname)-8s - [%(funcName)s:%(lineno)d] - %(message)s'
        logging.basicConfig(level=logging.INFO,
                            format=log_format,
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger(__name__)
        return logger
    

    def _user_set_logging_level(self) -> None:
        """Set logging level based on user configuration"""
        level = self._player_config.logging_lvl.upper()
        if level == "DEBUG":
            self._logger.setLevel(logging.DEBUG)
        elif level == "INFO":
            self._logger.setLevel(logging.INFO)
        elif level == "WARNING":
            self._logger.setLevel(logging.WARNING)
        elif level == "ERROR":
            self._logger.setLevel(logging.ERROR)
        elif level == "CRITICAL":
            self._logger.setLevel(logging.CRITICAL)
        else:
            self._logger.warning(f"Unknown logging level: {level}. Defaulting to INFO.")
            self._logger.setLevel(logging.INFO)


    def _read_sys_args_and_set_path(self) -> Path:
        """Read system arguments for configuration path, if no argument is given, use the default path

        Returns:
            Path: Path to configuration file
        """        
        global default_config_path_to_yaml
        """Read system arguments for configuration path"""
        if len(sys.argv) > 1:
            tempPath = Path(sys.argv[1])
        else:
            tempPath = default_config_path_to_yaml
        self._logger.info(f"The following config Path is used: {tempPath}")
        return tempPath


    def _open_config_file(self) -> dict:
        """Open the YAML configuration file, this file holds all settings for the configuration of the Dataset and Hardware

        Raises:
            FileExistsError: File/Path not found
            yaml.YAMLError: File could not be parsed

        Returns:
            dict: Loaded configuration as dictionary
        """
        try:
            with open(self._config_path.as_posix()  , 'r') as stream:
                configLoaded = yaml.safe_load(stream)
        except FileNotFoundError:
            self._logger.critical(f"The File/Path '{self._config_path}' are not found")
            raise FileExistsError(f"The File/Path '{self._config_path}' are not found")
        except yaml.YAMLError as e:
            self._logger.critical(f"Error parsing the YAML file: {e}")
            raise yaml.YAMLError(f"Error parsing the YAML file: {e}")
        else:
            self._logger.info("Configuration file loaded successfully")
            return configLoaded


    def _read_config(self) -> None:
        """Read and process the general configuration settings."""
        loaded_config = self._open_config_file()

        for key, value in loaded_config["Hardware"].items():
            if value["used"]:
                target_hardware = key
                if "output_open" in value:
                    output_open = value["output_open"]
                else:
                    output_open = None


        return DensppGenerationPlayerConfig(logging_lvl= loaded_config["General_Configuration"]["Logging_Lvl"],
                                            input= Path(loaded_config["Data_Input"]["input"]),
                                            target_hardware= target_hardware,
                                            output_open= output_open,
                                            start_time= loaded_config["Data_Configuration"]["Time"]["start_time"],
                                            end_time= loaded_config["Data_Configuration"]["Time"]["end_time"],
                                            do_cut= loaded_config["Data_Configuration"]["Data_Preprocessing"]["do_cut"],
                                            do_resample= loaded_config["Data_Configuration"]["Data_Preprocessing"]["do_resampling"],
                                            target_sampling_rate= loaded_config["Data_Configuration"]["Data_Preprocessing"]["sampling_rate_resample"],
                                            translation_value_voltage= loaded_config["Data_Configuration"]["Voltage_Scaling"]["translation_value_voltage"],
                                            channel_mapping= loaded_config["Data_Configuration"]["Hardware_Data_Mapping"]["channel_mapping"])


    def _config_data_translator(self) -> DataTranslator:
        """Configure the DataTranslator based on the current configuration

        Raises:
            Exception: Multiple hardware devices selected
            Exception: No hardware device selected 
            Exception: Specified hardware device not defined in output_devices.py

        Returns:
            DataTranslator: Configured hardware controller instance
        """
        # Load the specific device settings
        if self._player_config.target_hardware == "OscilloscopeMOX4":
            specific_device_settings = hardware_specification_oscilloscope_mox4(output_open= self._player_config.output_open)
        elif self._player_config.target_hardware == "DensPPPlayer":
            specific_device_settings = hardware_specification_denspp_player()
        elif self._player_config.target_hardware == "DensPPPlayer_import":
            specific_device_settings = hardware_specification_denspp_player_import()
        elif self._player_config.target_hardware == "DensPPPlayer_SDCard":
            specific_device_settings = hardware_specification_denspp_player_sdcard()
        else:
            self._logger.CRITICAL(f"The specified hardware device '{self._player_config.target_hardware}' is not defined in output_devices.py.")
            raise Exception(f"The specified hardware device '{self._player_config.target_hardware}' is not defined in output_devices.py.")

        data_translator = DataTranslator(specific_device_settings=specific_device_settings, # Create hardware controller with specific device settings
                                             logger=self._logger, 
                                             data_channel_mapping=self._player_config.channel_mapping)
        self._logger.info(f"Data Translator configured for device: {self._player_config.target_hardware}")
        return data_translator


    def _config_call_handler_SettingsData(self) -> SettingsData:
        """Configure the SettingsData based on the current configuration, This Object is used to load data in the ControllData class

        Returns:
            SettingsData: Configured SettingsData object
        """        
        deployed_settingsData = SettingsData(pipeline ="PipelineV0",
                                                do_merge = False,
                                                path = self._player_config.input,
                                                data_set = self._player_config.input.parts[-1],
                                                data_case = 0,
                                                data_point = 0,
                                                ch_sel = [],
                                                fs_resample = self._player_config.target_sampling_rate,
                                                t_range_sec = [self._player_config.start_time, self._player_config.end_time] if self._player_config.start_time is not None and self._player_config.end_time is not None else [],
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


    def _cut_data(self) -> None:
        """Cut data using the playerControllerData"""
        if self._player_config.do_cut:
            self._deployed_playerControllerData.do_cut()
            
            data = self._deployed_playerControllerData.get_data()
            
            if self._player_config.logging_lvl.upper() == "DEBUG":
                plot_data(data.data_raw[0,:], data.fs_used, data.time_end, "cutting")
            self._logger.info(f"Cut data with sampling rate {data.fs_used} with {len(data.data_raw)}")
        else:
            self._logger.info("Data cutting is disabled in the configuration.")


    def _resample_data(self) -> None:
        """Resample data to the desired sampling rate, defined in the yaml configuration file

        Raises:
            Exception: Desired resampling rate exceeds hardware capabilities
        """        
        if self._deployed_playerControllerData._settings.fs_resample > self._deployed_data_translator._dac_max_sampling_rate:# Check if the desired sampling rate is supported by the hardware
            self._logger.CRITICAL(f"The desired resampling rate of {self._deployed_playerControllerData._settings.fs_resample} Hz exceeds the maximum supported rate of {self._deployed_data_translator._dac_max_sampling_rate} Hz for the selected hardware.")
            raise Exception(f"The desired resampling rate of {self._deployed_playerControllerData._settings.fs_resample} Hz exceeds the maximum supported rate of {self._deployed_data_translator._dac_max_sampling_rate} Hz for the selected hardware.")

        if self._player_config.do_resample:
            self._deployed_playerControllerData.do_resample(num_points_mean=200)
            
            if self._player_config.logging_lvl.upper() == "DEBUG": # Plot data after resampling, if the logging level is set to DEBUG
                data = self._deployed_playerControllerData.get_data()
                plot_data(data.data_raw[0,:], data.fs_used, data.time_end, "resampling")
            
            self._logger.info(f"Data resampling completed, new sampling rate: {self._deployed_playerControllerData._raw_data.fs_used} Hz")
        else:
            self._logger.info("Data resampling is disabled in the configuration.")


    def _create_and_load_board_dataset_into_translator(self) -> None:       
        """Create and load the board dataset into the hardware settings controller"""
        data = self._deployed_playerControllerData.get_data()
        deployed_board_dataset = BoardDataset(data= data.data_raw,
                                                    samplingrate= data.fs_used,
                                                    groundtruth= [] if data.label_exist else None,
                                                    translation_value_voltage= self._player_config.translation_value_voltage)
        
        self._deployed_data_translator.load_data(deployed_board_dataset)
        self._logger.info("Board dataset loaded into hardware settings.")


    def _produce_data_for_hardware(self) -> None:
        """Produce data for the hardware device by translating it according to the device specifications."""
        self._deployed_data_translator.translation_for_device()
        self._logger.info("Data translation for hardware completed.")


    def _analyze_signals(self) -> None:
        """Analyze the original and processed signals"""
        processed_data = self._deployed_data_translator.get_data
        original_data_with_cut = self._untreated_raw_data_with_cut
        compartor = SignalCompartor(original_data_with_cut =original_data_with_cut.data_raw, 
                        signal_processed= processed_data.data, 
                        fs_original =original_data_with_cut.fs_orig, 
                        fs_processed= processed_data.samplingrate, 
                        scaling_factor= processed_data.translation_value_voltage)
        compartor.analyze_signals()
        results = compartor.get_results
        print(results)