import numpy as np
import csv
from dataclasses import dataclass

class Hardware_settings:
    # --- Settings of used DAC
    _dac_bit: int # Bits that the DAC can handle
    _dac_number_of_channels: int # Total number of channels the DAC channels 
    _dac_use_signed: bool # Whether the DAC uses signed values (e.g. +/- 5V or 0-10V)
    _dac_max_sampling_rate: int # Maximum sampling rate of the DAC
    _output_open: bool # Whether the output is open or 50 Ohm (needed for Oscilloscope)
    
    # --- Settings for dataset
    link_data2channel_num: list # Liste hat die Anzahl von Kanälen des DACs, es werden hier die Kanalnummern des Datensatzes angegeben, die auf den jeweiligen DAC Kanal gelegt werden sollen.

    
    # --- Settings for Groundtruth
    # Dies behinhalten exakte Spike-Zeitpunkte
    groundtruth_available: bool # Gibt an ob Groundtruth Daten im Datensatz vorhanden sind
    groundtruth_activate: list #Hat auch die Anzahl von Kanälen des DAC. Speichert Ture/False werte 
    # Ein True am Index 3 bedeutet: "Wenn auf dem vierten Hardware-Kanal ein Signal mit Ground-Truth-Daten abgespielt wird, dann aktiviere auch den dazugehörigen digitalen Trigger-Ausgang 

    logger: object
    def __init__(self, specific_device_settings: object, logger: object, data_channel_mapping: list) -> None:
        self.logger = logger
        self._dac_bit = specific_device_settings.verticalBit
        self._dac_number_of_channels = specific_device_settings.numChannels
        self._dac_use_signed = specific_device_settings.usedSigned
        self._dac_max_sampling_rate = specific_device_settings.max_sampling_rate
        if hasattr(specific_device_settings, 'output_open'):
            self._output_open = specific_device_settings.output_open
        
        self.data_channel_mapping = data_channel_mapping

        self.link_data2channel_num = [False for idx in range(self._dac_number_of_channels)]
        self.groundtruth_available = False
        self.groundtruth_activate = [False for _ in range(self._dac_number_of_channels)]

        self._set_channel_mapping()
        self._data = None

    @property
    def get_data (self):
        return self._data
    @property
    def dac_number_of_channels(self):
        return self._dac_number_of_channels

    @property
    def dac_bit(self):
        return self._dac_bit

    @property
    def dac_use_signed(self):
        return self._dac_use_signed
    
    @property
    def data_loaded(self) -> bool:
        """Check if data is loaded

        Returns:
            bool: True if data is loaded, False otherwise
        """        
        return True if self._data is not None else False
    
    def _set_channel_mapping(self) -> None:
        """Set the mapping from data channels to hardware channels"""        
        for data_channel, i in enumerate(self.data_channel_mapping):
            self.link_data2channel_num[i] = data_channel

    def translate_data_float2int(self, data_in: list) -> list:
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
            self.logger.warning("Overflow detected!")
            self.logger.warning(f"Max: {np.max(data_out[0])}, Min: {np.min(data_out[0])}")
        else:
            self.logger.info("No overflow detected in data conversion.")
        return data_out
    
    def translate_data_for_oscilloscope(self, resolution: float = 0.001) -> None:
        """Translate data to the Voltage range of the Oscillioscope"""        
        max_voltage_output = 10 if self._output_open else 5 # Max voltage depending on output mode (+/-10V open, +/-5V 50 Ohm)
        
        if self._data._translation_value_voltage is None:
            self.logger.info("No translation value is set, using the complete voltage range for scaling")
            max_range_use = True
        else:
            self.logger.info(f"Using translation value for voltage output: {self._data._translation_value_voltage}")
            max_range_use = False

        data_out = list()
        for data in self._data._data:
            abs_max_value_data = np.max(np.abs(data))
            
            # Multiply each value in the signal by the scaling factor.
            if max_range_use:
                scale_factor = max_voltage_output / abs_max_value_data
                scaled_signal = data * scale_factor
            elif not max_range_use:
                scaled_signal = data * self._data._translation_value_voltage
            
            scaled_signal = np.clip(scaled_signal, -max_voltage_output, max_voltage_output)
            
            # QUANTIZATION
                # 1. (divide): e.g., 3.14159V / 0.001 (resolution) = 3141.59
                # 2. (round): np.round(3141.59) = 3142
                # 3. (multiply): 3142 * 0.001 = 3.142
                # The result (3.142V) is a float, but it is on the 1mV grid.
            quantized_signal = np.round(scaled_signal / resolution) * resolution

            data_out.append(np.array(quantized_signal, dtype=np.float16))

        self._data._data = np.array(data_out, dtype=np.float16)
        self.logger.debug(f"Minimal Value: {np.min(quantized_signal)}, maximal Value: {np.max(quantized_signal)}")


    def create_csv_for_MXO4(self):
        """Output data in Oscilloscope MOX4 format"""
        with open('output_mox4.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Rate {self._data._samplingrate}"])
            for data in self._data._data[self.link_data2channel_num[0]]:
                    writer.writerow([data])

@dataclass
class Board_dataset:
    _data: np.ndarray #Saved the data to output to the hardware
    _samplingrate: float #Saved the sampling rate associated with the main data
    _groundtruth: list #Saved the trigger data associated with the main data
    _translation_value_voltage: float # Translation value from the data points to voltage output