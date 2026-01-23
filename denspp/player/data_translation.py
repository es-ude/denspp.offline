import numpy as np
import csv
from dataclasses import dataclass
from .output_devices import HardwareSpecifications

@dataclass
class BoardDataset:
    """Dataset structure for actual data to be output to hardware
    
    Attributes:
        data (np.ndarray): Data to be output to hardware
        samplingrate (float): Sampling rate of the data
        groundtruth (list): Ground truth events associated with the data
        translation_value_voltage (float): Translation value from data points to voltage output
    """
    data: np.ndarray #Saved the data to output to the hardware
    samplingrate: float #Saved the sampling rate associated with the main data
    groundtruth: list #Saved the trigger data associated with the main data
    translation_value_voltage: float # Translation value from the data points to voltage output


class DataTranslator:
    _logger: object # Logger from the main application
    _device_name: str # Name of the hardware device
    _dac_bit: int # Bits that the DAC can handle
    _dac_number_of_channels: int # Total number of channels the DAC channels 
    _dac_use_signed: bool # Whether the DAC uses signed values (e.g. +/- 5V or 0-10V)
    _dac_max_sampling_rate: int # Maximum sampling rate of the DAC
    _output_open: bool # Whether the output is open or 50 Ohm (needed for Oscilloscope)
    
    _link_data2channel_num: list # Mapping from data channels to hardware channels, e.g., [2,0,1] means data channel datachannel 2 goes to hardware channel 0, data channel 0 to hardware channel 1
    _data: BoardDataset # Data to be output to the hardware

    def __init__(self, specific_device_settings: HardwareSpecifications, logger: object, data_channel_mapping: list) -> None:
        """Initialize the DataTranslator with specific device settings, logger, and data channel mapping.

        Args:
            specific_device_settings (HardwareSpecifications): Hardware specifications for the device
            logger (object): Logger object for logging
            data_channel_mapping (list): Mapping from data channels to hardware channels
        """        
        self._logger = logger
        self._device_name = specific_device_settings.device_name
        self._dac_bit = specific_device_settings.verticalBit
        self._dac_number_of_channels = specific_device_settings.numChannels
        self._dac_use_signed = specific_device_settings.usedSigned
        self._dac_max_sampling_rate = specific_device_settings.max_sampling_rate
        self._output_open = specific_device_settings.output_open

        self._link_data2channel_num = [False for idx in range(self._dac_number_of_channels)]
        self._set_channel_mapping(data_channel_mapping)
        self._data = None


    # ========== API METHODS ==========
    @property
    def dac_number_of_channels(self) -> int:
        """Get the total number of DAC channels

        Returns:
            int: Number of DAC channels
        """        
        return self._dac_number_of_channels


    @property
    def dac_bit(self) -> int:
        """Get the bit depth of the DAC

        Returns:
            int: Bit depth of the DAC
        """        
        return self._dac_bit


    @property
    def dac_use_signed(self) -> bool:
        """Get whether the DAC uses signed values

        Returns:
            bool: True if DAC uses signed values, False otherwise
        """
        return self._dac_use_signed
    

    @property
    def data_loaded(self) -> bool:
        """Check if data is loaded

        Returns:
            bool: True if data is loaded, False otherwise
        """        
        return True if self._data is not None else False


    def load_data(self, board_dataset: BoardDataset) -> None:
        """Load data into the DataTranslator

        Args:
            board_dataset (BoardDataset): Data to be loaded
        """        
        self._data = board_dataset


    @property
    def get_data(self) -> BoardDataset:
        """Get the loaded data as BoardDataset

        Returns:
            BoardDataset: The loaded data
        """
        if self._data is None:
            self._logger.error("No data loaded in DataTranslator")
            return None
        else:        
            return self._data
        
    
    def translation_for_device(self) -> None:
        if self._device_name == "OscilloscopeMOX4":
            self._translate_data_for_oscilloscope()
            self._create_csv_for_MXO4()
        elif self._device_name == "DensPPPlayer":
            self._translate_data_for_oscilloscope(0.0001)
            self._create_csv_for_denspp_player()
        elif self._device_name == "DensPPPlayer_import":
            self._translate_data_for_oscilloscope(0.0001)
            self._translate_data_float2int()
        elif self._device_name == "DensPPPlayer_SDCard":
            self._translate_data_for_oscilloscope(0.0001)
            self._translate_data_float2int()
            self._create_csv_for_sd_card_denspp_player()
        else:
            raise ValueError(f"data_translation: {self._device_name} not implemnented yet")


    #  ========== INTERNAL METHODS ==========
    def _set_channel_mapping(self, data_channel_mapping: list) -> None:
        """Set the mapping from data channels to hardware channels"""        
        if len(data_channel_mapping) > self._dac_number_of_channels:
            self._logger.error("Data channel mapping length exceeds number of DAC channels")
            raise ValueError("Data channel mapping length exceeds number of DAC channels")
        for i in range(len(data_channel_mapping)):
            if data_channel_mapping[i] is not False:
                self._link_data2channel_num[i] = data_channel_mapping[i]


    def _translate_data_float2int(self, min_voltage: float = -5.0, max_voltage: float= 5.) -> None:
        data_out = list()
        for data in self._data.data:
            transformed_data_channel = []
            for data_point in data:   
                if data_point >= max_voltage:
                    val0 = max_voltage * (1 - 2**-15)
                elif data_point < min_voltage:
                    val0 = min_voltage
                else:
                    val0 = data_point
                transformed_data_channel.append(int(2**15 * (1 + val0 / max_voltage)))
            data_out.append(np.array(transformed_data_channel, dtype=np.uint16))
        self._data.data = np.array(data_out, dtype=np.uint16)
        

    def _translate_data_for_oscilloscope(self, resolution: float = 0.001) -> None:
        """Translate data to the Voltage range of the Oscillioscope"""
        if self._device_name == "OscilloscopeMOX4":
                max_voltage_output = 10 if self._output_open else 5 # Max voltage depending on output mode (+/-10V open, +/-5V 50 Ohm)
        else:
            max_voltage_output = 5 # Default value
        
        if self._data.translation_value_voltage is None:
            self._logger.info("No translation value is set, using the complete voltage range for scaling")
            max_range_use = True
        else:
            self._logger.info(f"Using translation value for voltage output: {self._data.translation_value_voltage}")
            max_range_use = False

        data_out = list()
        for data in self._data.data:
            abs_max_value_data = np.max(np.abs(data))
            
            # Multiply each value in the signal by the scaling factor.
            if max_range_use:
                scale_factor = max_voltage_output / abs_max_value_data
                scaled_signal = data * scale_factor
            elif not max_range_use:
                scaled_signal = data * self._data.translation_value_voltage
            
            scaled_signal = np.clip(scaled_signal, -max_voltage_output, max_voltage_output)
            
            # QUANTIZATION
                # 1. (divide): e.g., 3.14159V / 0.001 (resolution) = 3141.59
                # 2. (round): np.round(3141.59) = 3142
                # 3. (multiply): 3142 * 0.001 = 3.142
                # The result (3.142V) is a float, but it is on the 1mV grid.
            quantized_signal = np.round(scaled_signal / resolution) * resolution

            data_out.append(np.array(quantized_signal, dtype=np.float16))

        self._data.data = np.array(data_out, dtype=np.float16)
        self._logger.debug(f"Minimal Value: {np.min(quantized_signal)}, maximal Value: {np.max(quantized_signal)}")


    def _create_csv_for_MXO4(self) -> None:
        """Output data in Oscilloscope MOX4 format"""
        with open('output_mox4.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Rate {self._data.samplingrate}"])
            for data in self._data.data[self._link_data2channel_num[0]]:
                    writer.writerow([data])

    
    def _create_csv_for_denspp_player(self) -> None:
        """Output data in DensPP Player format"""        
        with open('output_denspp_player.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            if self._data.data.shape[0] < self._dac_number_of_channels:
                for _ in range(self._dac_number_of_channels - self._data.data.shape[0]):
                    self._data.data = np.vstack([self._data.data, np.zeros(self._data.data.shape[1])])
            num_samples = self._data.data.shape[1]
            for i in range(num_samples):
                row = [self._data.data[channel][i] if channel is not False else 0 for channel in self._link_data2channel_num]
                writer.writerow(row)


    def _create_csv_for_sd_card_denspp_player(self) -> None:
        with open("data_to_generate_denspp_player.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            if self._data.data.shape[0] < self._dac_number_of_channels:
                for _ in range(self._dac_number_of_channels - self._data.data.shape[0]):
                    self._data.data = np.vstack([self._data.data, np.zeros(self._data.data.shape[1])])
            num_samples = self._data.data.shape[1]
            for i in range(num_samples):
                row = [self._data.data[channel][i] if channel is not False else 0 for channel in self._link_data2channel_num]
                writer.writerow(row)