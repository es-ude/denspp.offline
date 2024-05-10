import dataclasses
import numpy as np
from package.data_process.process_noise import noise_real, noise_awgn, noise_flicker


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
