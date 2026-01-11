import numpy as np
import csv
from dataclasses import dataclass
from .output_devices import HardwareSpecifications

@dataclass
class BoardDataset:
    data: np.ndarray #Saved the data to output to the hardware
    samplingrate: float #Saved the sampling rate associated with the main data
    groundtruth: list #Saved the trigger data associated with the main data
    translation_value_voltage: float # Translation value from the data points to voltage output

class DataTranslator:
    _logger: object # Logger from the main application
    
    #Settings for the Output Device class
    _dac_bit: int # Bits that the DAC can handle
    _dac_number_of_channels: int # Total number of channels the DAC channels 
    _dac_use_signed: bool # Whether the DAC uses signed values (e.g. +/- 5V or 0-10V)
    _dac_max_sampling_rate: int # Maximum sampling rate of the DAC
    _output_open: bool # Whether the output is open or 50 Ohm (needed for Oscilloscope)
    
    _link_data2channel_num: list # Mapping from data channels to hardware channels, e.g., [2,0,1] means data channel datachannel 2 goes to hardware channel 0, data channel 0 to hardware channel 1
    _data: BoardDataset # Data to be output to the hardware

    def __init__(self, specific_device_settings: HardwareSpecifications, logger: object, data_channel_mapping: list) -> None:
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
        else:
            raise ValueError(f"data_translation: {self._device_name} not implemnented yet")


    #  ========== INTERNAL METHODS ==========
    def _set_channel_mapping(self, data_channel_mapping: list) -> None:
        """Set the mapping from data channels to hardware channels"""        
        for data_channel, i in enumerate(data_channel_mapping):
            self._link_data2channel_num[i] = data_channel


    def _translate_data_float2int(self, data_in: list) -> list:
        # max_dac_output define the Bit depth of the DAC 12bit -> 2^12 = 4096; signed -> 2048 
        """"""
        data_out = list()
        
        max_dac_output =(2** self._dac_bit) / 2 if self.dac_use_signed else (2** self._dac_bit)
        
        for data in data_in:
            # Find the absolute maximum value in the signal (regardless of positive or negative) 
            # With this call The loudest point in the signal (whether positive or negative) corresponds exactly to the target value max_dac_output.
            val_abs_max = data.max() if data.max() > abs(data.min()) else data.min()
            if val_abs_max < 0: # Necessary otherwise the Value get inverted
                val_abs_max = abs(val_abs_max)

            scaled_data = max_dac_output / val_abs_max * data
            if self._dac_use_signed:
                clipped_data = np.clip(scaled_data, -32768, 32767) # Prevent overflow for signed 16-bit integer
            else:
                clipped_data = np.clip(scaled_data, 0, 65535) # Prevent overflow for unsigned 16-bit integer

            data_out.append(np.array(clipped_data, dtype=np.int16)) # Convert to int16

        if np.any(data_out[0] > 32767) or np.any(data_out[0] < -32768):
            self._logger.warning("Overflow detected!")
            self._logger.warning(f"Max: {np.max(data_out[0])}, Min: {np.min(data_out[0])}")
        else:
            self._logger.info("No overflow detected in data conversion.")
        return data_out
    

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