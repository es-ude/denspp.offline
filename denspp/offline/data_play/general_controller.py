from logging import getLogger, Logger
from dataclasses import dataclass
from pathlib import Path

import yaml

from denspp.offline import get_path_to_project
from denspp.offline.data_call import SettingsData, ControllerData

from .data_translation import BoardDataset, DataTranslator
from .output_devices import (
    hardware_specification_player,
    hardware_specification_player_import,
    hardware_specification_player_sdcard,
    hardware_specification_mox4,
)
from .signal_validation import SignalComparator, SignalValidationResult


@dataclass
class DatasetReconstructionSettings:
    """Configuration for the General Player Controller

    Attributes:
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


class DatasetReconstruction:
    _logger: Logger
    _config_path: Path
    _player_config: DatasetReconstructionSettings
    _settings_data: SettingsData
    _loader_data: ControllerData
    _translator_data: DataTranslator

    _dataset_original: BoardDataset
    _dataset_deployed: BoardDataset

    def __init__(self, class_config: DatasetReconstructionSettings = None) -> None:
        """Init process for the configuration class"""
        self._logger = getLogger(__name__)
        self._config_path = Path(get_path_to_project("config")) / "hardware_config.yaml"

        if class_config is not None:
            self._player_config = class_config
        else:
            self._player_config = self._read_config()


    def load_data(self) -> None:
        """"""
        self._settings_data = self._config_call_handler_settings_data()
        self._translator_data = self._config_data_translator()

        #TODO: Check if this is works
        self._loader_data =
        sets = PlayerControllerData()
        sets._settings = self._settings_data
        sets.do_call()

        data = self._loader_data.get_data()
        if self._player_config.do_cut:
            self._loader_data.do_cut()

        if (
            self._loader_data._settings.fs_resample
            > self._translator_data._dac_max_sampling_rate
        ):  # Check if the desired sampling rate is supported by the hardware
            self._logger.critical(
                f"The desired resampling rate of {self._loader_data._settings.fs_resample} Hz exceeds the maximum supported rate of {self._translator_data._dac_max_sampling_rate} Hz for the selected hardware."
            )
            raise Exception(
                f"The desired resampling rate of {self._loader_data._settings.fs_resample} Hz exceeds the maximum supported rate of {self._translator_data._dac_max_sampling_rate} Hz for the selected hardware."
            )

        if self._player_config.do_resample:
            self._loader_data.do_resample(num_points_mean=200)

        self._dataset_original =
        self._create_and_load_board_dataset_into_translator()
        self._produce_data_for_hardware()

    @property
    def get_original_data(self) -> BoardDataset:
        """Output untreated raw data in BoardDataset format

        Returns:
            BoardDataset: Holdes untreated raw data, data name, data type, electrode id, fs original and fs used
        """
        return self._dataset_original

    @property
    def get_deployed_data(self) -> BoardDataset:
        """Output processed data in BoardDataset format

        Returns:
            BoardDataset: Holdes processed data and some metadata
        """
        return self._translator_data.get_data

    def _open_config_file(self) -> dict:
        """Open the YAML configuration file, this file holds all settings for the configuration of the Dataset and Hardware

        Raises:
            FileExistsError: File/Path not found
            yaml.YAMLError: File could not be parsed

        Returns:
            dict: Loaded configuration as dictionary
        """
        try:
            with open(self._config_path.as_posix(), "r") as stream:
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

    def _read_config(self) -> DatasetReconstructionSettings:
        """Read and process the general configuration settings."""
        loaded_config = self._open_config_file()

        target_hardware = None
        output_open = None
        for key, value in loaded_config["Hardware"].items():
            if value["used"]:
                target_hardware = key
                if "output_open" in value:
                    output_open = value["output_open"]
                else:
                    output_open = None

        return DatasetReconstructionSettings(
            input=Path(loaded_config["Data_Input"]["input"]),
            target_hardware=target_hardware,
            output_open=output_open,
            start_time=loaded_config["Data_Configuration"]["Time"]["start_time"],
            end_time=loaded_config["Data_Configuration"]["Time"]["end_time"],
            do_cut=loaded_config["Data_Configuration"]["Data_Preprocessing"]["do_cut"],
            do_resample=loaded_config["Data_Configuration"]["Data_Preprocessing"]["do_resampling"],
            target_sampling_rate=loaded_config["Data_Configuration"]["Data_Preprocessing"][
                "sampling_rate_resample"
            ],
            translation_value_voltage=loaded_config["Data_Configuration"]["Voltage_Scaling"][
                "translation_value_voltage"
            ],
            channel_mapping=loaded_config["Data_Configuration"]["Hardware_Data_Mapping"][
                "channel_mapping"
            ],
        )

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
            specific_device_settings = hardware_specification_mox4(
                output_open=self._player_config.output_open
            )
        elif self._player_config.target_hardware == "DensPPPlayer":
            specific_device_settings = hardware_specification_player()
        elif self._player_config.target_hardware == "DensPPPlayer_import":
            specific_device_settings = hardware_specification_player_import()
        elif self._player_config.target_hardware == "DensPPPlayer_SDCard":
            specific_device_settings = hardware_specification_player_sdcard()
        else:
            self._logger.critical(
                f"The specified hardware device '{self._player_config.target_hardware}' is not defined in output_devices.py."
            )
            raise Exception(
                f"The specified hardware device '{self._player_config.target_hardware}' is not defined in output_devices.py."
            )

        data_translator = DataTranslator(
            specific_device_settings=specific_device_settings,
            data_channel_mapping=self._player_config.channel_mapping,
        )
        self._logger.info(f"Data Translator configured for device: {self._player_config.target_hardware}")
        return data_translator

    def _config_call_handler_settings_data(self) -> SettingsData:
        """Configure the SettingsData based on the current configuration.
         This Object is used to load data in the ControllData class
        Returns:
            SettingsData: Configured SettingsData object
        """
        return SettingsData(
            pipeline="PipelineV0",
            do_merge=False,
            path="",
            data_set=self._player_config.input.parts[-1],
            data_case=0,
            data_point=0,
            ch_sel=[],
            fs_resample=self._player_config.target_sampling_rate,
            t_range_sec=[self._player_config.start_time, self._player_config.end_time]
            if self._player_config.start_time is not None and self._player_config.end_time is not None
            else [],
            do_mapping=False,
            is_mapping_str=False,
        )

    def _create_and_load_board_dataset_into_translator(self) -> None:
        """Create and load the board dataset into the hardware settings controller"""
        data = self._loader_data.get_data()
        deployed_board_dataset = BoardDataset(
            data=data.data_raw,
            samplingrate=data.fs_used,
            groundtruth=[] if data.label_exist else None,
            translation_value_voltage=self._player_config.translation_value_voltage,
        )

        self._translator_data.load_data(deployed_board_dataset)
        self._logger.info("Board dataset loaded into hardware settings.")

    def _produce_data_for_hardware(self) -> None:
        """Produce data for the hardware device by translating it according to the device specifications."""
        self._translator_data.translation_for_device()
        self._logger.info("Data translation for hardware completed.")

    def _analyze_signals(self) -> SignalValidationResult:
        """Analyze the original and processed signals"""
        processed_data = self.get_deployed_data
        original_data = self.get_original_data
        compartor = SignalComparator(
            original_data_with_cut=original_data.data,
            signal_processed=processed_data.data,
            fs_original=original_data.samplingrate,
            fs_processed=processed_data.samplingrate,
            scaling_factor=processed_data.translation_value_voltage,
        )
        compartor.analyze_signals()
        return compartor.get_results
