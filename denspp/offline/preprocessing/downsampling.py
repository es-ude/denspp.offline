import numpy as np
from dataclasses import dataclass


@dataclass
class SettingsDownSampling:
    """Settings class for configuring the properties of the downsampling module
    Attributes:
        sampling_rate:  Floating value with input sampling rate of the transient data stream
        dsr:            Integer with downsampling ratio for reducing the input sampling rate (SR_out = SR_in / OSR)
    """

    sampling_rate: float
    dsr: int


DefaultSettingsDownSampling = SettingsDownSampling(
    sampling_rate=1000.,
    dsr=10,
)


class DownSampling:
    def __init__(self, settings: SettingsDownSampling):
        self._settings = settings

    @property
    def sampling_rate_out(self) -> float:
        return self._settings.sampling_rate / self._settings.dsr

    def do_simple(self, uin: np.ndarray) -> np.ndarray:
        """Performing a simple downsampling of the adc data stream
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        n = uin.size // self._settings.dsr * self._settings.dsr
        data = uin[:n]
        return data.reshape(-1, self._settings.dsr).mean(axis=1)

    def do_cic(self, uin: np.ndarray, num_stages: int=5) -> np.ndarray:
        """Performing the CIC filter at the output of oversampled ADC
        param uin:          Numpy array with transient signal input (high sampling rate)
        param num_stages:   Number of stages to perform the CIC downsampling
        return:             Numpy array with transient signal output (low sampling rate)
        """
        output_transient = list()
        gain = (self._settings.dsr * 1) ** num_stages

        class integrator:
            def __init__(self):
                self.yn = 0
                self.ynm = 0

            def update(self, inp):
                self.ynm = self.yn
                self.yn = (self.ynm + inp)
                return (self.yn)

        class comb:
            def __init__(self):
                self.xn = 0
                self.xnm = 0

            def update(self, inp):
                self.xnm = self.xn
                self.xn = inp
                return (self.xn - self.xnm)

        intes = [integrator() for a in range(num_stages)]
        combs = [comb() for a in range(num_stages)]
        for (s, v) in enumerate(uin):
            z = v
            for i in range(num_stages):
                z = intes[i].update(z)

            if (s % self._settings.dsr) == 0:
                for c in range(num_stages):
                    z = combs[c].update(z)
                    j = z
                output_transient.append(j / gain)
        return np.array(output_transient)

    @staticmethod
    def do_decimation_polyphase_order_one(uin: np.ndarray) -> np.ndarray:
        """Performing first order Non-Recursive Polyphase Decimation on input
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        last_sample_hs = 0
        uout = []
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_hs)
            last_sample_hs = val

        uout = np.array(uout)
        return uout

    @staticmethod
    def do_decimation_polyphase_order_two(uin: np.ndarray) -> np.ndarray:
        """Performing second order Non-Recursive Polyphase Decimation on input
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        last_sample_hs = 0
        last_sample_ls = 0
        uout = []
        for idx, val in enumerate(uin):
            if idx % 2 == 1:
                uout.append(val + last_sample_ls + 2 * last_sample_hs)
                last_sample_ls = val
            last_sample_hs = val

        uout = np.array(uout)
        return uout

    def do_decimation_polyphase(self, uin: np.ndarray, take_first_order: bool=False) -> np.ndarray:
        """Performing Non-Recursive Polyphase Decimation on input (depends on DSR)
        param uin:          Numpy array with transient signal input (high sampling rate)
        return:             Numpy array with transient signal output (low sampling rate)
        """
        if self._settings.dsr % 2 == 0:
            raise ValueError("self._settings.dsr should be 2^x")

        x = uin
        for _ in range(int(np.log2(self._settings.dsr))):
            if take_first_order:
                x = self.do_decimation_polyphase_order_one(x)
            else:
                x = self.do_decimation_polyphase_order_two(x)
        return x
