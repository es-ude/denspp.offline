import numpy as np
from logging import getLogger
from dataclasses import dataclass
from scipy.constants import Boltzmann, elementary_charge


@dataclass
class SettingsNoise:
    """Settings for configuring the emulation of noise properties
    Attributes:
        temp:       Temperature [K]
        wgn_dB:     Effective spectral input noise power [dBW/sqrt(Hz)]
        Fc:         Corner frequency of the flicker (1/f) noise [Hz]
        slope:      Alpha coefficient of the flicker noise []
    """
    temp:       float
    wgn_dB:     float
    Fc:         float
    slope:      float

    @property
    def temp_celsius(self) -> float:
        return self.temp - 273.15

    @property
    def noise_pwr(self) -> float:
        return 4 * Boltzmann * self.temp_celsius

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self.temp / elementary_charge


DefaultSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6
)


class ProcessNoise:
    __print_device: str = ""

    def __init__(self, settings: SettingsNoise, fs_ana: float) -> None:
        """Processing analog noise for transient signals of electrical devices
        :param settings:    Dataclass for using noise simulation
        :param fs_ana:      Sampling frequency [Hz]
        """
        self.__settings_noise = settings
        self.__noise_sampling_rate = fs_ana
        self._logger = getLogger(__name__)

    ######################## FUNCTIONS FOR GENERATE NOISE ########################
    def __do_fft(self, y: np.ndarray) -> dict:
        window = np.hanning(y.size)
        fft_in = window * y
        N = y.size // 2
        fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
        fft_out[0] = fft_out[0] / 2
        freq = self.__noise_sampling_rate * np.fft.fftfreq(fft_out.size)

        # Taking positive range
        xsel = np.where(freq >= 0)
        fft_out = fft_out[xsel]
        freq = freq[xsel]
        return {'freq': freq, 'Y': fft_out}

    @staticmethod
    def __noise_awgn(size: int, fs: float, e_n: float) -> np.ndarray:
        """Generation of transient noise signal with spectral noise power [dBW/sqrt(Hz)]
        Args:
            e_n:    Spectral noise power density [dBW/sqrt(Hz)]
            fs:     Sampling rate
            size:   Number of points
        Returns:
            Numpy array with white gaussian noise signal
        """
        rho = 10 ** (e_n / 10)
        sigma = rho * np.sqrt(fs / 2)
        return np.random.randn(size) * sigma

    @staticmethod
    def __noise_flicker(size: int, alpha: float) -> np.ndarray:
        """Generating flicker noise
        Args:
            size:       no of points
            alpha:      Slope of the pink noise
        Returns:
            Numpy array with pink/flicker noise signal
        """
        window_size = 2 * size + (size % 2)
        # --- Thermal noise
        y_noise = np.random.randn(window_size)
        window = np.hanning(window_size)
        y_pink = np.fft.fft(window * y_noise)
        y_pink[0] = y_pink[0] / 2
        # --- Flicker spectral component
        n = np.arange(1, size + 1, 1)
        n = np.power(n, alpha)
        # --- Generate full spectrum
        y = y_pink[0:size] / n
        y_pos = y[np.newaxis, :]
        y_neg = np.fliplr(y_pos)
        y0 = np.concatenate((y_pos, y_neg), axis=None)
        # --- Generate pink noise
        u_pink = np.real(np.fft.ifft(y0)[0:size])
        u_pink -= np.mean(u_pink)
        return u_pink

    def __noise_real(self, tsize: int, fs: float, e_n: float, fc: float, alpha: float) -> np.ndarray:
        """Generation of real noise signal with specific noise power [dBW], corner frequency Fc, and slope of 1/f noise
        Args:
            tsize:      Size of time vector
            fs:         Sampling frequency [in Hz]
            e_n:        Spectral power density for the white gaussian noise generation [in dBW/sqrt(Hz)]
            fc:         Corner frequency of the white and pink noise
            alpha:      Slope of the pink noise
        Returns:
            Numpy array with real noise signal
        """
        # --- Generate noise components and match
        e_white = self.__noise_awgn(tsize, fs, e_n)
        e_pink = self.__noise_flicker(tsize, alpha)

        # --- Adapting the amplitude
        fft_white = self.__do_fft(e_white)
        fft_pink = self.__do_fft(e_pink)

        # --- Find corner frequency
        x_corner = np.argwhere(fft_white['freq'] >= fc)[0]
        n_mean = 100
        y_wgm = np.convolve(fft_white['Y'], np.ones(n_mean) / n_mean, mode='same')
        y_pnk = np.convolve(fft_pink['Y'], np.ones(n_mean) / n_mean, mode='same')
        scalef = y_wgm[x_corner] / y_pnk[x_corner]

        # --- Generate output noise
        return scalef * e_pink + e_white

    ######################## FUNCTIONS FOR HANDLING NOISE ########################
    @staticmethod
    def __calc_spectral_noise_device(dev_val: float) -> float:
        """Calculating the noise spectral density value"""
        assert dev_val > 0, "Apply only positive values"
        return 10 * np.log10(dev_val)

    def __calc_spectral_noise_pwr(self, fs: float) -> float:
        """Calculating the noise spectral density value"""
        val = self.__settings_noise.noise_pwr / fs
        return 3.01 + 10 * np.log10(val)

    def __calc_spectral_noise_volt(self, fs: float, resistance: float) -> float:
        """Calculating the noise spectral density value"""
        val = np.sqrt(2 * self.__settings_noise.noise_pwr / fs * resistance)
        return 10 * np.log10(val)

    def __calc_spectral_noise_curr(self, fs: float, resistance: float) -> float:
        """Calculating the noise spectral density value"""
        val = np.sqrt(2 * self.__settings_noise.noise_pwr / fs / resistance)
        return 3.01 + 10 * np.log10(val)

    @staticmethod
    def _calculate_params(noise_in: np.ndarray) -> [float, float]:
        """Calculating the parameters of effective input and std
        :param noise_in:    Numpy array of the noise input
        :return:            List with effective noise value and peak-to-peak noise value
        """
        noise_eff = np.std(noise_in)
        noise_pp = np.max(noise_in) - np.min(noise_in)
        return noise_eff, noise_pp

    def _do_print(self, noise_in: np.ndarray, mode_output: int=0) -> None:
        """Printing output from noise analysis
        :param noise_in:        Numpy array with generated noise signal
        :param mode_output:     Output mode [0: power in mW, 1: voltage in µV, 2: current in nA]
        """
        match mode_output:
            case 0:
                unit_text = "mW"
                unit_scale = 1e3
                unit_type = "power"
            case 1:
                unit_text = "µV"
                unit_scale = 1e6
                unit_type = "voltage"
            case 2:
                unit_text = "nA"
                unit_scale = 1e9
                unit_type = "current"
            case _:
                unit_text = ""
                unit_scale = 1e0
                unit_type = ""
        text_dev = f"" if len(self.__print_device) == 0 else f" ({self.__print_device})"

        noise_eff, noise_pp = self._calculate_params(noise_in)
        self._logger.debug(f"... effective input noise {unit_type}{text_dev}: {unit_scale * noise_eff:.4f} {unit_text}")
        self._logger.debug(f"... effective peak-to-peak noise {unit_type}{text_dev}: {unit_scale * noise_pp:.4f} {unit_text}")

    def gen_noise_real_pwr(self, size: int, e_n: float=0.0) -> np.ndarray:
        """Generating real transient noise power
        Args:
            size:       Number of iterations
            e_n:        Spectral power noise density [dBW/sqrt(Hz)]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__settings_noise.wgn_dB if e_n == 0.0 else e_n,
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 0)
        return u_noise

    def gen_noise_real_volt(self, size: int, resistance: float) -> np.ndarray:
        """Generating real transient noise voltage
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_volt(self.__noise_sampling_rate, resistance),
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 1)
        return u_noise

    def gen_noise_real_curr(self, size: int, resistance: float) -> np.ndarray:
        """Generating real transient noise current
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_curr(self.__noise_sampling_rate, resistance),
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 2)
        return u_noise

    def gen_noise_awgn_dev(self, size: int, dev_e: float) -> np.ndarray:
        """Generating white transient noise power
        Args:
            size:       Number of iterations
            dev_e:      Spectral noise voltage density from device [V/sqrt(Hz)]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_device(dev_e)
        )
        self._do_print(u_noise, 0)
        return u_noise

    def gen_noise_awgn_pwr(self, size: int, e_n: float=0.0) -> np.ndarray:
        """Generating white transient noise power
        Args:
            size:       Number of iterations
            e_n:        Spectral power noise density [dBW/sqrt(Hz)]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__settings_noise.wgn_dB if e_n == 0.0 else e_n,
        )
        self._do_print(u_noise, 0)
        return u_noise

    def gen_noise_awgn_volt(self, size: int, resistance: float) -> np.ndarray:
        """Generating white transient noise voltage
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_volt(self.__noise_sampling_rate, resistance)
        )
        self._do_print(u_noise, 1)
        return u_noise

    def gen_noise_awgn_curr(self, size: int, resistance: float) -> np.ndarray:
        """Generating white transient noise current
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = self.__noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_curr(self.__noise_sampling_rate, resistance)
        )
        self._do_print(u_noise, 2)
        return u_noise

    def gen_noise_flicker_volt(self, size: int) -> np.ndarray:
        """Generating white transient flicker noise voltage
        Args:
            size:       Number of iterations
            alpha:      Slope of flicker component in spectral part
        Returns:
            Numpy array with transient flicker noise
        """
        return self.__noise_flicker(size, self.__settings_noise.slope)
