import dataclasses
import numpy as np


@dataclasses.dataclass
class SettingsNoise:
    """Settings for configuring the pre-amp parasitics

    Inputs:
        wgndB:      Effective spectral input noise power [dBW/sqrt(Hz)]
        Fc:         Corner frequency of the flicker (1/f) noise [Hz]
        slope:      Alpha coefficient of the flicker noise []
        do_print:   Enable the noise output [True / False]
    """
    wgn_dB:     float
    Fc:         float
    slope:      float
    do_print:   bool


RecommendedSettingsNoise = SettingsNoise(
    wgn_dB=-100,
    Fc=10,
    slope=0.6,
    do_print=False
)


class ProcessNoise:
    """Processing analog noise for transient signals of electrical devices"""
    __print_device: str

    def __init__(self, settings: SettingsNoise, fs_ana: float):
        self.__settings_noise = settings
        self.__noise_sampling_rate = fs_ana

    def _gen_noise_real(self, size: int) -> np.ndarray:
        """Generating transient noise (real)"""
        # --- Generating noise
        u_noise, noise_eff_out, noise_pp = noise_real(
            tsize=size, fs=self.__noise_sampling_rate,
            wgndBW=self.__settings_noise.wgn_dB,
            Fc=self.__settings_noise.Fc,
            alpha=self.__settings_noise.slope
        )
        # --- Print output
        if self.__settings_noise.do_print:
            addon = f"" if len(self.__print_device) == 0 else f" ({self.__print_device})"
            print(f"... effective input noise{addon}: {1e6 * noise_eff_out:.4f} µV")
            print(f"... effective peak-to-peak noise{addon}: {1e6 * noise_pp:.4f} µV")

        return u_noise

    def _gen_noise_awgn(self, size: int) -> np.ndarray:
        """Generating transient noise ()"""
        # --- Generating noise
        u_noise, noise_eff_out, noise_pp = noise_awgn(
            size=size, fs=self.__noise_sampling_rate,
            wgndBW=self.__settings_noise.wgn_dB
        )
        # --- Print output
        if self.__settings_noise.do_print:
            addon = f"" if len(self.__print_device) == 0 else f" ({self.__print_device})"
            print(f"... effective input noise{addon}: {1e6 * noise_eff_out:.4f} µV")
            print(f"... effective peak-to-peak noise{addon}: {1e6 * noise_pp:.4f} µV")

        return u_noise

    def _gen_noise_flicker(self, size: int) -> np.ndarray:
        """Generating transient noise (flicker)"""
        # --- Generating noise
        u_noise, noise_eff_out, noise_pp = noise_flicker(
            size=size, alpha=self.__settings_noise.slope
        )
        # --- Print output
        if self.__settings_noise.do_print:
            addon = f"" if len(self.__print_device) == 0 else f" ({self.__print_device})"
            print(f"... effective input noise{addon}: {1e6 * noise_eff_out:.4f} µV")
            print(f"... effective peak-to-peak noise{addon}: {1e6 * noise_pp:.4f} µV")

        return u_noise


def noise_awgn(size: int, fs: float, wgndBW: float) -> [np.ndarray, np.ndarray]:
    """Generation of transient noise signal with noise power [dB]
    input:
    wgndB   - spectral noise density [dBW/SQRT(Hz)]
    fs      - sample rate
    size    - no of points
    output:
    noise   - N points of noise signal with spectral noise density of rho
    n_eff   - effective value of output noise
    """
    rho = 10 ** (wgndBW / 10)
    sigma = rho * np.sqrt(fs / 2)
    noise = np.random.randn(size) * sigma

    # Calculation of effective noise
    noise_eff = np.std(noise)
    noise_pp = np.max(noise) - np.min(noise)
    return noise, noise_eff


def noise_flicker(size: int, alpha: float) -> np.ndarray:
    """Generating flicker noise"""
    M = 2 * size + (size % 2)
    # --- Thermal noise
    ynoise = np.random.randn(M)
    window = np.hanning(M)
    Ypink = np.fft.fft(window * ynoise)
    Ypink[0] = Ypink[0] / 2
    # --- Flicker spectral component
    n = np.arange(1, size + 1, 1)
    n = np.power(n, alpha)
    # --- Generate full spectrum
    Y = Ypink[0:size] / n
    Ypos = Y[np.newaxis, :]
    Yneg = np.fliplr(Ypos)
    Y = np.concatenate((Ypos, Yneg), axis=None)
    # --- Generate pink noise
    U = np.fft.ifft(Y)
    Upink = np.real(U[0:size])
    Upink -= np.mean(Upink)
    # Output
    return Upink


def noise_real(tsize: int, fs: float, wgndBW: float, Fc: float, alpha: float) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Generation of real noise signal with specific noise power [dBW], corner frequency Fc, and slope of 1/f noise"""
    # --- Generate noise components and match
    Uwhite, _ = noise_awgn(tsize, fs, wgndBW)
    Upink = noise_flicker(tsize, alpha)

    # --- Adapting the amplitude
    freq0, Ywhite = do_fft(Uwhite, fs)
    _, Ypink = do_fft(Upink, fs)

    # --- Find corner frequency
    X_Fc = np.argwhere(freq0 >= Fc)
    X_Fc = X_Fc[0]
    Nmean = 100
    YWm = np.convolve(Ywhite, np.ones(Nmean) / Nmean, mode='same')
    YPm = np.convolve(Ypink, np.ones(Nmean) / Nmean, mode='same')
    scalef = YWm[X_Fc] / YPm[X_Fc]

    # --- Generate output noise
    Unoise = scalef * Upink + Uwhite
    noise_eff = np.std(Unoise)
    noise_pp = np.max(Unoise) - np.min(Unoise)

    return Unoise, noise_eff, noise_pp


# -------- TEST ROUTINE ------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from package.signal_analyse import do_fft

    fs = 2e3
    t = np.arange(0, 2e6, 1) / fs

    # Real Noise
    noise_pink = noise_awgn(t.size, fs, -130)[0]
    noise_f = noise_flicker(t.size, 1)
    noise_out = noise_real(t.size, fs, -130, 1, 0.9)[0]
    freq1, psd_real = do_fft(noise_out, fs)

    scale = 1e3
    plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)

    ax1.set_title('Thermal noise')
    ax1.plot(t, scale * noise_f)
    ax4.hist(scale * noise_f, bins=100, density=True)
    ax7.psd(scale * noise_f, Fs=fs)

    ax2.set_title('Flicker noise')
    ax2.plot(t, scale * noise_pink)
    ax5.hist(scale * noise_pink, bins=100, density=True)
    ax8.psd(scale * noise_pink, Fs=fs)

    ax3.set_title('Real noise')
    ax3.plot(t, scale * noise_out)
    ax6.hist(scale * noise_out, bins=100, density=True)
    ax9.loglog(freq1, psd_real)

    plt.tight_layout()

    # ---- Noise plot
    plt.figure()
    A = np.random.randint(-5, 5, 1000)
    plt.hist(A, bins=100)

    plt.tight_layout()

    # --- Show all plots
    plt.show()
