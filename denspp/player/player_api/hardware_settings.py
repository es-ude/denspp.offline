import numpy as np
import csv

class Hardware_settings:
    # --- Settings of used DAC
    _dac_bit: int # Bits that the DAC can handle
    _dac_use_channel: int # Number of channels used
    _dac_use_signed: bool # Whether the DAC uses signed values (e.g. +/- 5V or 0-10V)
    _dac_max_sampling_rate: int # Maximum sampling rate of the DAC
    _output_open: bool # Whether the output is open or 50 Ohm (needed for Oscilloscope)
    
    # --- Settings for dataset
    link_data2channel_num: list # Liste hat die Anzahl von Kanälen des DACs, es werden hier die Kanalnummern des Datensatzes angegeben, die auf den jeweiligen DAC Kanal gelegt werden sollen.
    link_data2channel_use: list # Liste hat die Anzahl von Kanälen des DACs, gibt an ob der Kanal verwendet wird oder nicht
    
    # --- Settings for Groundtruth
    # Dies behinhalten exakte Spike-Zeitpunkte
    groundtruth_available: bool # Gibt an ob Groundtruth Daten im Datensatz vorhanden sind
    groundtruth_activate: list #Hat auch die Anzahl von Kanälen des DAC. Speichert Ture/False werte 
    # Ein True am Index 3 bedeutet: "Wenn auf dem vierten Hardware-Kanal ein Signal mit Ground-Truth-Daten abgespielt wird, dann aktiviere auch den dazugehörigen digitalen Trigger-Ausgang 

    logger: object
    def __init__(self, specific_device_settings: object, logger: object) -> None:
        self.logger = logger
        self._dac_bit = specific_device_settings.verticalBit
        self._dac_use_channel = specific_device_settings.numChannels
        self._dac_use_signed = specific_device_settings.usedSigned
        self._dac_max_sampling_rate = specific_device_settings.max_sampling_rate
        if hasattr(specific_device_settings, 'output_open'):
            self._output_open = specific_device_settings.output_open

        self.link_data2channel_use = [False for _ in range(self._dac_use_channel)]
        self.link_data2channel_num = [idx for idx in range(self._dac_use_channel)]
        self.groundtruth_available = False
        self.groundtruth_activate = [False for _ in range(self._dac_use_channel)]

    @property
    def dac_use_channel(self):
        return self._dac_use_channel

    @property
    def dac_bit(self):
        return self._dac_bit

    @property
    def dac_use_signed(self):
        return self._dac_use_signed

    def check_active_channel(self) -> None:
        """"""
        for idx, ch in enumerate(self.link_data2channel_num):
            self.link_data2channel_use[idx] = True if ch else False


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
    
    def translate_data_for_oscilloscope(self, data_in: list) -> list:
        """Translate data for the Oscilloscope"""
        max_voltage_output = 10 if self._output_open else 5 # Max voltage depending on output mode (+/-10V open, +/-5V 50 Ohm)
        resolution = 0.001

        data_out = list()
        for data in data_in:
            abs_max_value_data = np.max(np.abs(data))
            scale_factor = max_voltage_output / abs_max_value_data
            
            # Multiply each value in the signal by the scaling factor.
            scaled_signal = data * scale_factor
            scaled_signal = np.clip(scaled_signal, -max_voltage_output, max_voltage_output)
            
            # QUANTIZATION
                # 1. (divide): e.g., 3.14159V / 0.001 (resolution) = 3141.59
                # 2. (round): np.round(3141.59) = 3142
                # 3. (multiply): 3142 * 0.001 = 3.142
                # The result (3.142V) is a float, but it is on the 1mV grid.
            quantized_signal = np.round(scaled_signal / resolution) * resolution

            data_out.append(np.array(quantized_signal, dtype=np.float16))

            self.logger.debug(f"Minimal Value: {np.min(quantized_signal)}, maximal Value: {np.max(quantized_signal)}")
        return data_out


class Board_dataset:
    # --- Selected data to set
    data: list #Saved the data to output to the hardware
    samplingrate: float #Saved the sampling rate associated with the main data
    groundtruth: list #Saved the trigger data associated with the main data
    # --- Data and GPIO for streaming to DAC (w/o pre-processing)

    def __init__(self) -> None:
        self.data = []
        self.samplingrate = 0.0
        self.groundtruth = []
    
    def create_csv_for_MXO4(self):
        """Output data in Oscilloscope MOX4 format"""
        with open('output_mox4.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow([f"Rate {self.samplingrate}"])
            for idx, data in enumerate(self.data):
                for value in data:
                    writer.writerow([value])





def create_data_empty(point: int, dac_bit: int, dac_signed=False) -> list:
    """
        List with two numpy arrays
    """
    offset = 0 if dac_signed else 1
    stream_data = np.zeros(shape=(point,), dtype=np.int16) + offset * np.power(2, dac_bit - 1)
    stream_trg = np.zeros(shape=(point,), dtype=bool)
    return stream_data, stream_trg


def create_data_dummy_sine(point: int, dac_bit: int, dac_signed=False) -> list:
    """
    Return:
        List with two numpy arrays
    """
    offset = 0 if dac_signed else 1
    stream_data = np.zeros(shape=(point,), dtype=np.int16)
    stream_trg = np.zeros(shape=(point,), dtype=bool)

    # --- Apply the sinusoida waveform to output
    for idx in range(point):
        stream_data[idx] = np.power(2, dac_bit - 1) * (offset + 0.98 * np.sin(2 * np.pi * idx / point))

    # --- Apply max peak to trigger output
    pos = np.argwhere(stream_data == stream_data.max())
    stream_trg[pos] = True
    pos = np.argwhere(stream_data == stream_data.min())
    stream_trg[pos] = True
    
    # --- Return data
    return stream_data, stream_trg