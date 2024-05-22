import dataclasses
import numpy as np
from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsCUR:
    """Individual data class to configure the current amplifier

    Args:
        vdd:            Positive supply voltage [V]
        vss:            Negative supply voltage [V]
        fs_ana:         Sampling frequency of input [Hz]
        transimpedance: Transimpedance value [V/A]
        offset_v:       Offset voltage of current amplifier [V]
        offset_i:       Offset current of current amplifier [A]
        noise_en:       Enable noise on output [True / False]
        para_en:        Enable parasitic [True / False]
    """
    vdd:            float
    vss:            float
    fs_ana:         float
    # Amplifier characteristics
    transimpedance: float
    offset_v:       float
    offset_i:       float
    # Settings for parasitic
    noise_en:       bool
    para_en:        bool

    @property
    def vcm(self) -> float:
        return (self.vdd + self.vss) / 2


RecommendedSettingsCUR = SettingsCUR(
    vdd=0.9, vss=-0.9,
    fs_ana=50e3,
    transimpedance=1e3,
    offset_v=1e-6, offset_i=1e-12,
    noise_en=False,
    para_en=False
)


class CurrentAmplifier(ProcessNoise):
    """Class for emulating an analogue current amplifier"""
    _settings_noise: SettingsNoise
    __print_device = "current amplifier"

    def __init__(self, settings_dev: SettingsCUR, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings_dev.fs_ana)
        self._settings = settings_dev

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self._settings.vdd] = self._settings.vdd
        uin[uin < self._settings.vss] = self._settings.vss
        return uin

    def __gen_noise(self, size: int) -> np.ndarray:
        """Generating noise"""
        return self._gen_noise_awgn(size, True)

    def __add_parasitic(self, size: int) -> np.ndarray:
        """"""
        u_para = np.zeros((size, ))
        u_para += self._settings.transimpedance * self._settings.offset_i
        u_para += self._settings.offset_v
        u_para += self._settings.vcm
        # Adding noise
        if self._settings.noise_en:
            u_para += self._gen_noise_real(size)

        return u_para

    def transimpedance_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the transimpedance amplifier (single, normal) with input signal

        Args:
            iin:    Input current [A]
            uinn:   Negative input voltage [V]
        Returns:
            Test signal
        """
        u_out = self._settings.transimpedance * iin
        u_out += self.__add_parasitic(u_out.size)
        return self.__voltage_clipping(u_out)

    def push_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS push/source current amplifier"""
        u_out = np.zeros(iin.shape)
        x_neg = np.argwhere(iin < 0)
        u_out[x_neg,] = iin[x_neg,] * self._settings.transimpedance
        u_out += self.__add_parasitic(u_out.size)
        return self.__voltage_clipping(u_out)

    def pull_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS pull/sink current amplifier"""
        u_out = np.zeros(iin.shape)
        x_pos = np.argwhere(iin >= 0)
        u_out[x_pos, ] = iin[x_pos, ] * self._settings.transimpedance
        u_out += self.__add_parasitic(u_out.size)
        return self.__voltage_clipping(u_out)

    def push_pull_amplifier(self, iin: np.ndarray) -> [np.ndarray, np.ndarray]:
        """Performing the CMOS push-pull current amplifier"""
        u_pos = self.pull_amplifier(iin)
        u_neg = self.push_amplifier(iin)
        return u_pos, u_neg

    def push_pull_abs_amplifier(self, iin: np.ndarray) -> np.ndarray:
        """Performing the CMOS push-pull current absolute amplifier"""
        u_out = self.pull_amplifier(iin)
        u_out -= self.push_amplifier(iin)
        return u_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # --- Definiton of signals
    t_end = 50e-3
    fs = 10e3
    i_off = 0.0
    i_pp = [1e-3, 10e-6]
    f_sig = [100.0, 250.0]

    tA = np.linspace(0, t_end, int(t_end * fs), endpoint=True)
    i_in = np.zeros(tA.shape) + i_off
    for idx, f0 in enumerate(f_sig):
        i_in += i_pp[idx] * np.sin(2 * np.pi * f0 * tA)

    # --- Data Processing
    dev_cur = CurrentAmplifier(RecommendedSettingsCUR)

    u_out = list()
    u_out.append(i_in)
    u_out.append(dev_cur.transimpedance_amplifier(i_in))
    u_out.append(dev_cur.push_amplifier(i_in))
    u_out.append(dev_cur.pull_amplifier(i_in))
    u_out.append(dev_cur.push_pull_amplifier(i_in)[0])
    u_out.append(dev_cur.push_pull_amplifier(i_in)[1])
    u_out.append(dev_cur.push_pull_abs_amplifier(i_in))

    u_dict = ["I_in", "Transimpedance", "Push", "Pull", "Push-Pull (Pos.)", "Push-Pull (Neg.)", "Push-Pull (Abs.)"]

    # --- Plotten
    plt.figure()
    plt_size = [2, 4]
    axs = [plt.subplot(plt_size[0], plt_size[1], 1+idx) for idx in range(plt_size[0] * plt_size[1])]

    for idx, u_sig in enumerate(u_out, start=0):
        axs[idx].plot(tA, u_sig, 'k')
        axs[idx].grid()
        axs[idx].set_title(u_dict[idx])

    axs[6].set_xlabel("Time t / s")
    plt.tight_layout()
    plt.show()
