import dataclasses

@dataclasses.dataclass
class OscilloscopeMOX4:
    """Settings for the Oscilloscope MOX4 hardware"""
    verticalBit: int = 16  # Bit depth of the DAC
    numChannels: int = 1  # Number of DAC channels
    max_sampling_rate: int = 625e6  # Maximum sampling rate in Hz
    usedSigned: bool = True  # Whether the Hardware uses signed values

    output_open: bool =False  # If the output is open (+/-10V) or 50 Ohm (+/-5V)