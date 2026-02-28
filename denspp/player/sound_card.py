import numpy as np
import soundcard as sc
from denspp.offline.plot_helper import get_plot_color


class TranslatorSoundCard:
    __audio_name: str
    __samp_rate: int
    __mic: None
    __snd: None

    def __init__(self, name: str="", samp_rate: float=44.1e3) -> None:
        """Class for Getting and Pushing audio signals into a sound card
        :param name:        Name of the sound card
        :param samp_rate:   Floating with Sampling rate [Hz]
        :return:            None
        """
        self.__audio_name = name
        self.__samp_rate = int(samp_rate)

    @property
    def get_bitwidth(self) -> int:
        """Returning the bitwidth of the sound card"""
        return 16

    @property
    def get_voltage_range(self) -> tuple[float, float]:
        """Returning the voltage range of the sound card"""
        return -1.5, 1.5

    @property
    def get_voltage_resolution(self) -> float:
        """Returning the voltage resolution of the sound card"""
        return (self.get_voltage_range[1] - self.get_voltage_range[0]) / 2 ** self.get_bitwidth

    @property
    def get_sampling_rate(self) -> float:
        """Returning the sampling rate applied to the sound card"""
        return self.__samp_rate

    @property
    def get_list_available_speaker(self) -> list:
        """Getting a list with all available speaker devices"""
        return sc.all_speakers()

    @property
    def get_list_available_microphones(self) -> list:
        """Getting a list with all available microphone devices"""
        return sc.all_microphones()

    def get_speaker(self):
        """Returning the used speaker device"""
        if self.__audio_name == "":
            self.__snd = sc.default_speaker()
        else:
            raise NotImplementedError
        return self.__snd

    def get_microphone(self):
        """Returning the used microphone device"""
        if self.__audio_name == "":
            self.__mic = sc.default_microphone()
        else:
            raise NotImplementedError
        return self.__mic

    def update_sampling_rate(self, samp_rate: float) -> None:
        """Function for updating the sampling rate of the sound card
        :param samp_rate:   Floating with Sampling rate [Hz]
        :return:            None
        """
        self.__samp_rate = int(samp_rate)

    def play_data(self, signal_in: np.ndarray) -> None:
        """Function for playing the signal into the speaker connectors of the sound card device
        :param signal_in:   Numpy array with signal input
        :return:            None
        """
        if not len(signal_in.shape) == 1:
            raise NotImplementedError

        if self.__snd is None:
            self.get_speaker()

        self.__snd.play(
            data=signal_in,
            samplerate=self.__samp_rate,
            channels=[0]
        )

    def get_data(self, duration_sec: float) -> np.ndarray:
        """Function for getting signal data, recorded from the microphone connector
        :param duration_sec:    Floating with duration [s]
        :return:                Numpy array with 1D-transient signal data
        """
        if self.__mic is None:
            self.get_microphone()

        return self.__mic.record(
            numframes=int(self.__samp_rate * duration_sec)+1,
            samplerate=self.__samp_rate,
            channels=[0],
            blocksize=None
        ).flatten()

    def plot_data(self, data: np.ndarray) -> None:
        """"Plotting the recorded data from device
        :param data:    Numpy array with signal data
        :return:        None
        """
        import matplotlib.pyplot as plt
        time = np.linspace(start=0, stop=data.shape[0], num=data.shape[0], endpoint=False) / self.__samp_rate

        plt.figure()
        plt.plot(time, data, color=get_plot_color(0), label=f"CH #{0}")

        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.xlim([0, time[-1]])
        plt.grid(True)

        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    dut = TranslatorSoundCard()
    dut.get_speaker()
    dut.get_microphone()
    do_play = True
    window = 10.

    if not do_play:
        a = dut.get_data(duration_sec=window)
        dut.plot_data(a)
    else:
        time = np.linspace(start=0, stop=window, num=int(window * dut.get_sampling_rate))
        signal = np.sin(2*np.pi*time*200.) + np.random.randn(*time.shape) * 0.01
        dut.play_data(signal)
