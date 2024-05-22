import dataclasses
import numpy as np
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsINT:
    """Individual data class to configure an analog voltage integrator

    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        fs_ana:     Sampling frequency of input [Hz]
        gain:       Amplification [V/V]
        offset_v:     Offset voltage of the amplifier [V]
        noise_en:   Enable noise on output [True / False]
        para_en:    Enable parasitic [True / False]
    """
    vdd:        float
    vss:        float
    fs_ana:     float
    # Amplifier characteristics
    gain:       int
    offset_v:   float
    offset_i:   float
    noise_en:   bool
    para_en:    bool

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsINT = SettingsINT(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3, gain=40,
    offset_v=1e-6,
    offset_i=1e-12,
    noise_en=False,
    para_en=False
)


class VoltageIntegrator(ProcessNoise):
    """Class for emulating an analogue voltage integrator"""
    _settings_noise: SettingsNoise
    __print_device = "voltage integrator"

    def __init__(self, settings_dev: SettingsINT, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self._settings.vdd] = self._settings.vdd
        uin[uin < self._settings.vss] = self._settings.vss
        return uin

    def do(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs the pre-amplification (single, normal) with input signal

        Args:
            uinp:   Positive input voltage [V]
            uinn:   Negative input voltage [V]

        Returns:
            Test signal
        """
        du = uinp - uinn
        u_out = du
        u_out += self._settings.gain * self._settings.offset_v
        u_out += self._settings.vcm

        # Adding noise
        if self._settings.noise_en:
            u_out += self._settings.gain * self._gen_noise_real(du.size)

        return self.__voltage_clipping(u_out)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dev_test = VoltageIntegrator(RecommendedSettingsINT)

    # --- Definition of Inputs
    f_smp = 10e3
    t_end = 0.2
    u_off = 0e-3
    upp = [0.1, 0.01]
    f0 = [100, 500]

    time = np.linspace(0, t_end, int(t_end * f_smp), endpoint=True)
    u_inp = np.zeros(time.shape) + u_off
    for idx, peak_value in enumerate(upp):
        u_inp += peak_value * np.sin(2 * np.pi * time * f0[idx])
    u_inn = np.array(RecommendedSettingsINT.vcm)

    # --- DUT
    u_out = dev_test.do(u_inp, u_inn)

    # --- Test condition (here, Figure)
    plt.figure()
    plt.plot(time, u_inp, 'k', label="input")
    plt.plot(time, u_out, 'r', label="output")
    plt.grid()
    plt.xlabel("Time t / s")
    plt.ylabel("Voltage U_x / V")

    plt.tight_layout()
    plt.show()




