import dataclasses
import numpy as np

from package.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


@dataclasses.dataclass
class SettingsSTIM:
    """"""
    update_rate: float
    use_noise: bool


class StimModulator(ProcessNoise):
    """"""

    def __init__(self, settings: SettingsSTIM, settings_noise=RecommendedSettingsNoise):
        super().__init__(settings_noise, settings.update_rate)
        self._settings_stim = settings
        self._settings_noise = settings_noise

    def __generate_noise(self, size: int, voltage_out=True) -> np.ndarray:
        return self._gen_noise_awgn(size, voltage_out)

    def do_current_stimulation(self, amplitude: float, signal: np.ndarray) -> np.ndarray:
        """"""
        return amplitude * signal

    def do_voltage_stimulation(self, amplitude: float, signal: np.ndarray) -> np.ndarray:
        """"""
        return amplitude * signal

    def do_charge_stimulation(self, amplitude: float, signal: np.ndarray) -> np.ndarray:
        """"""
        return amplitude * signal


RecommendedSettingsStim = SettingsSTIM(
    update_rate=20e3,
    use_noise=False
)


if __name__ == "__main__":
    pass
