import dataclasses
import numpy as np
from package.signal_analyse import do_fft


@dataclasses.dataclass
class SettingsNoise:
    """Settings for configuring the pre-amp parasitics
    Args:
        wgn_dB:     Effective spectral input noise power [dBW/sqrt(Hz)]
        Fc:         Corner frequency of the flicker (1/f) noise [Hz]
        slope:      Alpha coefficient of the flicker noise []
        do_print:   Enable the noise output [True / False]
    """
    wgn_dB:     float
    Fc:         float
    slope:      float
    do_print:   bool


RecommendedSettingsNoise = SettingsNoise(
    wgn_dB=-120,
    Fc=10,
    slope=0.6,
    do_print=True
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

    def _do_print(self, noise_in: np.ndarray, volt_output=True) -> None:
        """Printing output from noise analysis"""
        if self.__settings_noise.do_print:
            unit_text = "µV" if volt_output else "nA"
            unit_scale = 1e6 if volt_output else 1e9
            addon = f"" if len(self.__print_device) == 0 else f" ({self.__print_device})"

            noise_eff, noise_pp = _calculate_params(noise_in)
            print(f"... effective input noise{addon}: {unit_scale * noise_eff:.4f} {unit_text}")
            print(f"... effective peak-to-peak noise{addon}: {unit_scale * noise_pp:.4f} {unit_text}")

    def _gen_noise_real(self, size: int, volt_output=True) -> np.ndarray:
        """Generating transient noise (real)"""
        u_noise = noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            wgn_pwr=self.__settings_noise.wgn_dB,
            fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, volt_output)

        return u_noise

    def _gen_noise_awgn(self, size: int, volt_output=True) -> np.ndarray:
        """Generating transient noise ()"""
        u_noise = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            wgn_pwr=self.__settings_noise.wgn_dB
        )
        self._do_print(u_noise, volt_output)

        return u_noise

    def _gen_noise_flicker(self, size: int, volt_output=True) -> np.ndarray:
        """Generating transient noise (flicker)"""
        # --- Generating noise
        u_noise = noise_flicker(
            size=size, alpha=self.__settings_noise.slope
        )
        self._do_print(u_noise, volt_output)

        return u_noise


def noise_awgn(size: int, fs: float, wgn_pwr: float) -> np.ndarray:
    """Generation of transient noise signal with noise power [dB]
    Args:
        wgn_pwr:    spectral noise density [dBW/SQRT(Hz)]
        fs:         sample rate
        size:       no of points
    Returns:
        Numpy array with white gaussian noise signal
    """
    rho = 10 ** (wgn_pwr / 10)
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


def noise_real(tsize: int, fs: float, wgn_pwr: float, fc: float, alpha: float) -> np.ndarray:
    """Generation of real noise signal with specific noise power [dBW], corner frequency Fc, and slope of 1/f noise
    Args:
        tsize:      Size of time vector
        fs:         Sampling frequency [in Hz]
        wgn_pwr:    Power for the white gaussian noise generation [in dBW]
        fc:         Corner frequency of the white and pink noise
        alpha:      Slope of the pink noise
    Returns:
        Numpy array with real noise signal
    """
    # --- Generate noise components and match
    u_white = noise_awgn(tsize, fs, wgn_pwr)
    u_pink = noise_flicker(tsize, alpha)

    # --- Adapting the amplitude
    freq0, y_white = do_fft(u_white, fs)
    _, y_pink = do_fft(u_pink, fs)

    # --- Find corner frequency
    x_corner = np.argwhere(freq0 >= fc)[0]
    n_mean = 100
    y_wgm = np.convolve(y_white, np.ones(n_mean) / n_mean, mode='same')
    y_pnk = np.convolve(y_pink, np.ones(n_mean) / n_mean, mode='same')
    scalef = y_wgm[x_corner] / y_pnk[x_corner]

    # --- Generate output noise
    return scalef * u_pink + u_white


# -------- TEST ROUTINE ------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fs0 = 2e3
    t = np.arange(0, 2e6, 1) / fs0

    # Real Noise
    noise_pink = noise_awgn(t.size, fs0, -130)
    noise_f = noise_flicker(t.size, 1)
    noise_out = noise_real(t.size, fs0, -130, 1, 0.9)
    freq1, psd_real = do_fft(noise_out, fs0)

    plt.close('all')
    # --- Plotting
    scale = 1e3
    plt.figure()
    axs = [plt.subplot(3, 3, idx+1) for idx in range(9)]

    axs[0].set_title('Thermal noise')
    axs[0].plot(t, scale * noise_f)
    axs[3].hist(scale * noise_f, bins=100, density=True)
    axs[6].psd(scale * noise_f, Fs=fs0)

    axs[1].set_title('Flicker noise')
    axs[1].plot(t, scale * noise_pink)
    axs[4].hist(scale * noise_pink, bins=100, density=True)
    axs[7].psd(scale * noise_pink, Fs=fs0)

    axs[2].set_title('Real noise')
    axs[2].plot(t, scale * noise_out)
    axs[5].hist(scale * noise_out, bins=100, density=True)
    axs[8].loglog(freq1, psd_real)

    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.show()
