import dataclasses
import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from package.data_process.transformation import do_fft


@dataclasses.dataclass
class SettingsNoise:
    """Settings for configuring the pre-amp parasitics
    Args:
        temp:       Temperature [K]
        wgn_dB:     Effective spectral input noise power [dBW/sqrt(Hz)]
        Fc:         Corner frequency of the flicker (1/f) noise [Hz]
        slope:      Alpha coefficient of the flicker noise []
        do_print:   Enable the noise output [True / False]
    """
    temp:       float
    wgn_dB:     float
    Fc:         float
    slope:      float
    do_print:   bool

    @property
    def temp_celsius(self) -> float:
        return self.temp - 273.15

    @property
    def noise_pwr(self) -> float:
        return 4 * Boltzmann * self.temp_celsius

    @property
    def temperature_voltage(self) -> float:
        return Boltzmann * self.temp / elementary_charge


RecommendedSettingsNoise = SettingsNoise(
    temp=300,
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=False
)


def _calculate_params(noise_in: np.ndarray) -> [float, float]:
    """Calculating the parameters of effective input and std"""
    noise_eff = np.std(noise_in)
    noise_pp = np.max(noise_in) - np.min(noise_in)
    return noise_eff, noise_pp


class ProcessNoise:
    """Processing analog noise for transient signals of electrical devices"""
    __print_device = ""

    def __init__(self, settings: SettingsNoise, fs_ana: float):
        self.__settings_noise = settings
        self.__noise_sampling_rate = fs_ana

    @staticmethod
    def __calc_spectral_noise_device(dev_val: float) -> float:
        """Calculating the noise spectral density value"""
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

    def _do_print(self, noise_in: np.ndarray, mode_output=0) -> None:
        """Printing output from noise analysis"""
        if self.__settings_noise.do_print:
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

            noise_eff, noise_pp = _calculate_params(noise_in)
            print(f"... effective input noise {unit_type}{text_dev}: {unit_scale * noise_eff:.4f} {unit_text}")
            print(f"... effective peak-to-peak noise {unit_type}{text_dev}: {unit_scale * noise_pp:.4f} {unit_text}")

    def _gen_noise_real_pwr(self, size: int) -> np.ndarray:
        """Generating real transient noise power
        Args:
            size:       Number of iterations
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__settings_noise.wgn_dB,
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 0)
        return u_noise

    def _gen_noise_real_volt(self, size: int, resistance: float) -> np.ndarray:
        """Generating real transient noise voltage
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_volt(self.__noise_sampling_rate, resistance),
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 1)
        return u_noise

    def _gen_noise_real_curr(self, size: int, resistance: float) -> np.ndarray:
        """Generating real transient noise current
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_curr(self.__noise_sampling_rate, resistance),
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, 2)
        return u_noise

    def _gen_noise_awgn_dev(self, size: int, dev_e: float) -> np.ndarray:
        """Generating white transient noise power
        Args:
            size:       Number of iterations
            dev_e:      Spectral noise voltage density from device
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_device(dev_e)
        )
        self._do_print(u_noise, 0)
        return u_noise

    def _gen_noise_awgn_pwr(self, size: int) -> np.ndarray:
        """Generating white transient noise power
        Args:
            size:       Number of iterations
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__settings_noise.wgn_dB
        )
        self._do_print(u_noise, 0)
        return u_noise

    def _gen_noise_awgn_volt(self, size: int, resistance: float) -> np.ndarray:
        """Generating white transient noise voltage
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_volt(self.__noise_sampling_rate, resistance)
        )
        self._do_print(u_noise, 1)
        return u_noise

    def _gen_noise_awgn_curr(self, size: int, resistance: float) -> np.ndarray:
        """Generating white transient noise current
        Args:
            size:       Number of iterations
            resistance: Resistance value [Ohm]
        Returns:
            Numpy array with transient noise
        """
        u_noise = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            e_n=self.__calc_spectral_noise_curr(self.__noise_sampling_rate, resistance)
        )
        self._do_print(u_noise, 2)
        return u_noise


def noise_awgn(size: int, fs: float, e_n: float) -> np.ndarray:
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


def noise_flicker(size: int, alpha: float) -> np.ndarray:
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


def noise_real(tsize: int, fs: float, e_n: float, fc: float, alpha: float) -> np.ndarray:
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
    e_white = noise_awgn(tsize, fs, e_n)
    e_pink = noise_flicker(tsize, alpha)

    # --- Adapting the amplitude
    freq0, y_white = do_fft(e_white, fs)
    _, y_pink = do_fft(e_pink, fs)

    # --- Find corner frequency
    x_corner = np.argwhere(freq0 >= fc)[0]
    n_mean = 100
    y_wgm = np.convolve(y_white, np.ones(n_mean) / n_mean, mode='same')
    y_pnk = np.convolve(y_pink, np.ones(n_mean) / n_mean, mode='same')
    scalef = y_wgm[x_corner] / y_pnk[x_corner]

    # --- Generate output noise
    return scalef * e_pink + e_white


# -------- TEST ROUTINE ------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # --- Signal generation
    fs0 = 2e3
    t = np.arange(0, 2e7, 1) / fs0
    noise_w = noise_awgn(t.size, fs0, -70)
    noise_f = noise_flicker(t.size, 0.9)
    noise_r = noise_real(t.size, fs0, -70, 100, 0.6)

    # --- Plotting
    scale = [1e3, 1e3, 1e3]
    plt.close('all')
    plt.figure()
    axs = [plt.subplot(3, 3, idx+1) for idx in range(9)]

    axs[0].set_title('Flicker noise')
    axs[0].plot(t, scale[0] * noise_f)
    axs[3].hist(scale[0] * noise_f, bins=100, density=True)
    freq0, psd_real0 = do_fft(noise_f, fs0)
    axs[6].loglog(freq0, psd_real0)

    axs[1].set_title('White noise')
    axs[1].plot(t, scale[1] * noise_w)
    axs[4].hist(scale[1] * noise_w, bins=100, density=True)
    freq1, psd_real1 = do_fft(noise_w, fs0)
    axs[7].loglog(freq1, psd_real1)

    axs[2].set_title('Real noise')
    axs[2].plot(t, scale[2] * noise_r)
    axs[5].hist(scale[2] * noise_r, bins=100, density=True)
    freq2, psd_real2 = do_fft(noise_r, fs0)
    axs[8].loglog(freq2, psd_real2)

    # --- Labeling
    for ax in axs:
        ax.grid()

    for idx in range(0, 3):
        axs[idx].set_xlabel('Time t / s')
        axs[idx].set_ylabel('Voltage U / mV')

    for idx in range(6, 9):
        axs[idx].set_xlabel('Frequency f / Hz')
        axs[idx].set_ylabel('e_n / V/sqrt(Hz)')

    plt.tight_layout()
    plt.show()
