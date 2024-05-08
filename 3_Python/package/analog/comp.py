import dataclasses
import numpy as np
from package.data_process.process_noise import noise_awgn


@dataclasses.dataclass
class SettingsCMP:
    """"Individual data class to configure an analogue comparator

    Args:
        vdd:        Positive supply voltage [V]
        vss:        Negative supply voltage [V]
        offset:     Offset voltage of the amplifier [V]
        noise:      Enable noise on output [True/False]
    """
    vdd:    float
    vss:    float
    # Comparator characteristics
    gain: int
    out_analog: bool
    offset: float
    noise:  bool

    @property
    def vcm(self) -> np.ndarray:
        return np.array(self.vdd + self.vss) / 2


RecommendedSettingsAMP = SettingsCMP(
    vdd=0.6, vss=-0.6,
    out_analog=True,
    gain=10,
    offset=1e-3,
    noise=True
)


class Comp:
    """Class for emulating an analogue comparator"""
    settings: SettingsCMP
    _unoise: np.ndarray
    _int_state: np.ndarray

    def __init__(self, setting: SettingsCMP):
        # --- Settings
        self.settings = setting

    def __gen_noise(self, input: int, scale=0.1) -> np.ndarray:
        """Generate the input noise"""
        return np.random.normal(self.settings.offset, scale, input) if self.settings.noise else self.settings.offset * np.ones(input)

    def __voltage_clipping(self, uin: np.ndarray) -> np.ndarray:
        """Do voltage clipping at voltage supply"""
        uin[uin > self.settings.vdd] = self.settings.vdd
        uin[uin < self.settings.vss] = self.settings.vss
        return uin

    def __cmp_calc(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performing the difference calculation"""
        du = uinp - uinn
        return self.settings.vcm + self.settings.gain * du

    def __dig_output(self, du: np.ndarray) -> np.ndarray:
        """Translating the analogue signal into digital trigger"""
        return np.sign(du) == 1

    def cmp_ideal(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs an ideal comparator with input signal
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        return self.__voltage_clipping(u_cmp)

    def cmp_normal(self, uinp: np.ndarray, uinn: np.ndarray) -> np.ndarray:
        """Performs a normal comparator with input signal
        Args:
            uinp    - Positive input voltage [V]
            uinn    - Negative input voltage [V]
        """
        u_cmp = self.__cmp_calc(uinp, uinn)
        self._unoise = self.__gen_noise(u_cmp.size, 1e-1)
        return self.__voltage_clipping(u_cmp - self._unoise)

    def cmp_single_pos_hystere(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr=0.25) -> np.ndarray:
        """Performs a single-side hystere comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        """
        du = uinp - uinn
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        self._unoise = self.__gen_noise(du.size, 1e-3)
        u_out = np.zeros(du.shape)

        # state == 0 --> not active, state == 1 --> active
        dthr0 = [scale_thr * (self.settings.vdd - self.settings.vcm), 0.0]
        for idx, val in enumerate(du):
            out = val - (dthr0[0] if not self._int_state[idx] else dthr0[1]) + self._unoise[idx]
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1

            u_out[idx] = self.settings.vcm + self.settings.gain * out

        return self.__voltage_clipping(u_out)

    def cmp_single_neg_hystere(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr=0.25) -> np.ndarray:
        """Performs a single-side hystere comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        """
        du = uinp - uinn
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        self._unoise = self.__gen_noise(du.size, 1e-3)
        u_out = np.zeros(du.shape)

        # state == 0 --> not active, state == 1 --> active
        dthr0 = [0.0, scale_thr * (self.settings.vss - self.settings.vcm)]
        for idx, val in enumerate(du):
            out = val - (dthr0[0] if not self._int_state[idx] else dthr0[1]) + self._unoise[idx]
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1

            u_out[idx] = self.settings.vcm + self.settings.gain * out

        return self.__voltage_clipping(u_out)

    def cmp_double_hystere(self, uinp: np.ndarray, uinn: np.ndarray, scale_thr=0.25) -> np.ndarray:
        """Performs a double-side hystere comparator with input signal
        Args:
            uinp: Positive input voltage [V]
            uinn: Negative input voltage [V]
            scale_thr: Scaling value from supply voltage [default = 25%]
        """
        du = uinp - uinn
        self._int_state = np.zeros(du.shape, dtype=np.bool_)
        self._unoise = self.__gen_noise(du.size, 1e-3)
        u_out = np.zeros(du.shape)

        # state == 0 --> not active, state == 1 --> active
        dthr0 = [scale_thr * (self.settings.vdd - self.settings.vcm), scale_thr * (self.settings.vss - self.settings.vcm)]
        for idx, val in enumerate(du):
            out = val - (dthr0[0] if not self._int_state[idx] else dthr0[1]) + self._unoise[idx]
            if idx < u_out.size-2:
                self._int_state[idx+1] = np.sign(out) == 1

            u_out[idx] = self.settings.vcm + self.settings.gain * out

        return self.__voltage_clipping(u_out)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("TEST")

    cmp = Comp(RecommendedSettingsAMP)
    # --- Defining the input
    n_samples = 100000
    inp0 = np.concatenate((np.linspace(-0.6, 0.6, n_samples), np.linspace(0.6, -0.6, n_samples)))
    inp1 = np.zeros((n_samples, ))

    # --- Defining the output
    vcm = cmp.settings.vcm
    out0 = cmp.cmp_single_neg_hystere(inp0, vcm)
    out1 = cmp.cmp_normal(inp1, vcm)

    # --- Plots
    plt.figure(1)
    axs = [plt.subplot(2, 1, idx+1) for idx in range(2)]

    axs[0].plot(inp0, out0)
    axs[0].grid()
    axs[1].set_ylabel("Output voltage [V]")

    axs[1].hist(cmp._unoise, bins=100, align="left", label="output")
    print(np.median(cmp._unoise))
    axs[1].grid()
    axs[1].legend()
    axs[1].set_xlabel("Input voltage [V]")

    plt.tight_layout()
    plt.show()
