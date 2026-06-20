from dataclasses import dataclass
from .devices.sound_card import TranslatorSoundCard


@dataclass
class HardwareSpecifications:
    """Hardware Settings for deploying / reconstructing pre-recorded data into player hardware
    Attributes:
        device_name (str): Name of the hardware device
        verticalBit (int): Bit depth of the DAC
        numChannels (int): Number of DAC channels
        max_sampling_rate (float): Maximum sampling rate in Hz
        usedSigned (bool): Whether the Hardware uses signed values
        output_open (bool): If the output is open (+/-10V) or 50 Ohm (+/-5V)
    """
    device_name: str
    verticalBit: int
    numChannels: int
    max_sampling_rate: float
    usedSigned: bool
    output_open: bool


def hardware_specification_mox4(output_open: bool) -> HardwareSpecifications:
    """Setup the hardware specifications for the Oscilloscope MOX4

    Args:
        output_open (bool): Whether the output is open (+/-10V) or 50 Ohm (+/-5V)

    Returns:
        HardwareSpecifications: Hardware specifications for the Oscilloscope MOX4
    """
    return HardwareSpecifications(
        device_name="OscilloscopeMOX4",
        verticalBit=16,
        numChannels=1,
        max_sampling_rate=625e6,
        usedSigned=True,
        output_open=output_open,
    )


def hardware_specification_player() -> HardwareSpecifications:
    """Setup the hardware specifications for the denspp.player

    Returns:
        HardwareSpecifications: Hardware specifications for the denspp.player
    """
    return HardwareSpecifications(
        device_name="DensPPPlayer",
        verticalBit=16,
        numChannels=4,
        max_sampling_rate=1e6,
        usedSigned=True,
        output_open=False,
    )


def hardware_specification_player_sdcard() -> HardwareSpecifications:
    """Setup the hardware specifications for the denspp.player, for the setup to safe the data on an SD Card

    Returns:
        HardwareSpecifications: Hardware specifications for the denspp.player SD Card setup
    """
    return HardwareSpecifications(
        device_name="DensPPPlayer_SDCard",
        verticalBit=16,
        numChannels=4,
        max_sampling_rate=1e6,
        usedSigned=True,
        output_open=False,
    )


def hardware_specification_player_import() -> HardwareSpecifications:
    """Setup the hardware specifications for the denspp.player, for the setup to import data

    Returns:
        HardwareSpecifications: Hardware specifications for the DensPP Player import setup
    """
    return HardwareSpecifications(
        device_name="DensPPPlayer_import",
        verticalBit=16,
        numChannels=4,
        max_sampling_rate=1e6,
        usedSigned=True,
        output_open=False,
    )


def hardware_specification_soundcard() -> HardwareSpecifications:
    """Setup the hardware specifications for the SoundCard Import """
    dut = TranslatorSoundCard()
    return HardwareSpecifications(
        device_name="SoundCard",
        verticalBit=dut.get_bitwidth,
        numChannels=1,
        max_sampling_rate=dut.get_sampling_rate,
        usedSigned=True,
        output_open=False,
    )
