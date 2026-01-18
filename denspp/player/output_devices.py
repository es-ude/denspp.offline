from dataclasses import dataclass

@dataclass
class HardwareSpecifications:
    """Hardware Settingns
    
    Attributes:
        device_name (str): Name of the hardware device
        verticalBit (int): Bit depth of the DAC
        numChannels (int): Number of DAC channels
        max_sampling_rate (int): Maximum sampling rate in Hz
        usedSigned (bool): Whether the Hardware uses signed values
        output_open (bool): If the output is open (+/-10V) or 50 Ohm (+/-5V)
    """
    device_name: str 
    verticalBit: int
    numChannels: int
    max_sampling_rate: int
    usedSigned: bool
    output_open: bool

def hardware_specification_oscilloscope_mox4() -> HardwareSpecifications:
    """Setup the hardware specifications for the Oscilloscope MOX4

    Returns:
        HardwareSpecifications: Hardware specifications for the Oscilloscope MOX4
    """    
    return HardwareSpecifications(
        device_name="OscilloscopeMOX4",
        verticalBit=16,
        numChannels=1,
        max_sampling_rate=625e6,
        usedSigned=True,
        output_open=False
    )


def hardware_specification_denspp_player() -> HardwareSpecifications:

    return HardwareSpecifications(
        device_name="DensPPPlayer",
        verticalBit=16,
        numChannels=4,
        max_sampling_rate=1e6,
        usedSigned=True,
        output_open=None
    )