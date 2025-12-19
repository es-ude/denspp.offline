import numpy as np
from dataclasses import dataclass
from logging import getLogger, Logger
from scipy import signal
from fxpmath import Fxp, Config
from denspp.offline import check_keylist_elements_any
from denspp.offline.analog import ProcessNoise, SettingsNoise, DefaultSettingsNoise


@dataclass(frozen=True)
class WaveformSignal:
    """Dataclass with waveform signal
    Attributes:
        time:   Numpy array with timestamps
        signal: Numpy array with signal
        fs:     Float with sampling rate
        rms:    Float with root mean square value
    """
    time: np.ndarray
    signal: np.ndarray
    fs: float
    rms: float

class WaveformGenerator:
    __handler_noise: ProcessNoise
    _logger: Logger

    def __init__(self, sampling_rate: float, add_noise: bool=False, settings_noise: SettingsNoise=DefaultSettingsNoise):
        """Class for generating the transient stimulation signal
        :param sampling_rate:   Sampling rate of the signal
        :param add_noise:       Boolean for adding noise to output
        :param settings_noise:  Settings noise to add to output
        """
        self._logger = getLogger(__name__)
        self.__handler_noise = ProcessNoise(settings_noise, sampling_rate)
        self.__add_noise: bool = add_noise
        self._sampling_rate: float = sampling_rate
        self._time_duration: float = 1.0

        self.__func_dict = {'RECT_HALF': self.__generate_rectangular_half}
        self.__func_dict.update({'RECT_FULL': self.__generate_rectangular_full})
        self.__func_dict.update({'LIN_RISE': self.__generate_linear_rising})
        self.__func_dict.update({'LIN_FALL': self.__generate_linear_falling})
        self.__func_dict.update({'SINE_HALF': self.__generate_sinusoidal_half})
        self.__func_dict.update({'SINE_HALF_INV': self.__generate_sinusoidal_half_inverse})
        self.__func_dict.update({'SINE_FULL': self.__generate_sinusoidal_full})
        self.__func_dict.update({'TRI_HALF': self.__generate_triangle_half})
        self.__func_dict.update({'TRI_FULL': self.__generate_triangle_full})
        self.__func_dict.update({'SAW_POS': self.__generate_sawtooth_positive})
        self.__func_dict.update({'SAW_NEG': self.__generate_sawtooth_negative})
        self.__func_dict.update({'GAUSS': self.__generate_gaussian})
        self.__func_dict.update({'ZERO': self.__generate_zero})
        self.__func_dict.update({'EAP': self.__generate_spike_waveform})

    @property
    def _num_samples(self) -> int:
        """Calculating the number of samples of the transient window"""
        return int(self._time_duration * self._sampling_rate)

    @property
    def _build_time_cycle(self) -> np.ndarray:
        return np.linspace(start=0.0, stop=2 * np.pi, num=self._num_samples, endpoint=False, dtype=float)

    @staticmethod
    def __switching_polarity(signal_in: np.ndarray, do_cathodic: bool) -> np.ndarray:
        """Switching the polarity for cathodic-first (True) or anodic-first (False) waveform"""
        return signal_in if not do_cathodic else (-1) * signal_in

    def __get_charge_balancing_factor(self, waveforms: list) -> float:
        """Getting the coefficient for area-related comparison for charge balancing the biphasic waveform"""
        if not len(waveforms) == 2 and not len(waveforms) == 3:
            self._logger.info("It is not a biphasic waveform available - Please check!")
            return 1.0
        else:
            area_first = np.trapezoid(waveforms[0])
            area_second = np.trapezoid(waveforms[-1])
            return np.abs(area_first / area_second)

    def check_charge_balancing(self, signal: np.ndarray) -> float:
        """Checking if stimulation signal is charge balanced"""
        dq = np.trapezoid(signal)
        self._logger.info(f"... waveform has an error of {dq:.6f}")
        return dq

    def __generate_zero(self) -> np.ndarray:
        out = np.zeros((self._num_samples,), dtype=float)
        return out

    def __generate_rectangular_half(self) -> np.ndarray:
        return 1.0 + self.__generate_zero()

    def __generate_rectangular_full(self) -> np.ndarray:
        return signal.square(self._build_time_cycle, duty=0.5)

    def __generate_linear_rising(self) -> np.ndarray:
        return np.linspace(0.0, 1.0,  self._num_samples, endpoint=True, dtype=float)

    def __generate_linear_falling(self) -> np.ndarray:
        return np.linspace(1.0, 0.0, self._num_samples, endpoint=True, dtype=float)

    def __generate_sinusoidal_half(self) -> np.ndarray:
        return np.sin(0.5* self._build_time_cycle, dtype=float)

    def __generate_sinusoidal_half_inverse(self) -> np.ndarray:
        return 1.0 - np.sin(0.5* self._build_time_cycle, dtype=float)

    def __generate_sinusoidal_full(self) -> np.ndarray:
        return np.sin(self._build_time_cycle, dtype=float)

    def __generate_triangle_half(self) -> np.ndarray:
        return signal.sawtooth(0.5*self._build_time_cycle + np.pi/2, width=0.5)

    def __generate_triangle_full(self) -> np.ndarray:
        return signal.sawtooth(self._build_time_cycle + np.pi/2, width=0.5)

    def __generate_sawtooth_positive(self) -> np.ndarray:
        return 2 * self.__generate_linear_rising() - 1.0

    def __generate_sawtooth_negative(self) -> np.ndarray:
        return 2 * self.__generate_linear_falling() - 1.0

    def __generate_gaussian(self) -> np.ndarray:
        time = self.__generate_sawtooth_positive()
        out = signal.gausspulse(time, fc=np.pi, retenv=True)[1]
        scale_amp = (out.max()+out.min())/(out.max())
        return out * scale_amp - out.min()

    def __generate_spike_waveform(self) -> np.ndarray:
        t_end_ms = 1.6
        t = np.linspace(start=0., stop=t_end_ms, num=int(t_end_ms * self._sampling_rate * 1e-3), endpoint=True)
        eap0 = -np.exp(-((t - 0.45) ** 2) / 0.03)
        eap1 = 0.5 * np.exp(-((t - 0.86) ** 2) / 0.08)
        eap = eap0 + eap1
        eap = -eap / eap.min()
        return eap

    def get_dictionary_classes(self) -> list:
        """Getting a list with class names / labels of waveforms
        :return:            List with class names
        """
        return [val for val in self.__func_dict.keys()]

    def __select_waveform_template(self, time_duration: float, sel_wfg: str, do_cathodic: bool=False) -> np.ndarray:
        """Selection for generating a waveform template
        Args:
            time_duration:  Time window for the waveform
            sel_wfg:        Selected waveform type [0: rect., 1: linear-rising, 2: linear-falling, 3: half-sinusoidal,
                            4: half-sinusoidal (inverse), 5: full-sinusoidal, 6: half-triangular, 7: full-triangular,
                            8: positive sawtooth, 9: negative sawtooth, 10: gaussian]
            do_cathodic:    Boolean for cathodic-first impulse
        Returns:
            Numpy array with selected waveform
        """
        if sel_wfg in self.__func_dict.keys():
            self._time_duration = time_duration
            signal = self.__func_dict[sel_wfg]()
            waveform = self.__switching_polarity(signal, do_cathodic)
            self._logger.debug(f"Selected waveform type {sel_wfg} is generated with shape {waveform.shape}")
            return waveform
        else:
            raise NotImplementedError("Waveform is not implemented!")

    def generate_noise(self, shape: tuple) -> np.ndarray:
        """Generating a transient signal with noise
        :param shape:   Numpy shape of the transient signal
        :return:        Numpy array with noise signal
        """
        if len(shape) == 2:
            noise = np.zeros(shape)
            for dim0 in range(shape[0]):
                noise[dim0,:] = self.__handler_noise.gen_noise_real_pwr(shape[1])
        else:
            noise = self.__handler_noise.gen_noise_real_pwr(shape[0])
        return noise

    def generate_waveform(self, time_points: list, time_duration: list,
                          waveform_select: list, polarity_cathodic: list) -> WaveformSignal:
        """Generating the signal with waveforms for stimulation
        :param time_points:         List of time points for applying a stimulation waveform
        :param time_duration:       List of stimulation waveform duration
        :param waveform_select:     List of selected waveforms
        :param polarity_cathodic:   List for performing cathodic-first generation
        :returns:                   Dataclass WaveformSignal with numpy arrays ['time', output_signal, true rms value)
        """
        if not len(time_points) == len(waveform_select) == len(time_duration):
            raise RuntimeError("Please check input! --> Length is not equal")
        else:
            self._time_duration = 2 * time_points[-1] + time_duration[-1]
            out = self.__generate_zero()
            rms_value = 0.0
            for idx, (time_off, time_sec, wvf_type) in enumerate(zip(time_points, time_duration, waveform_select)):
                time_xpos = int(time_off * self._sampling_rate)
                do_polarity = polarity_cathodic[idx] if not len(polarity_cathodic) == 0 else False
                waveform = self.__select_waveform_template(time_sec, wvf_type, do_polarity)
                out[time_xpos:time_xpos+waveform.size] += waveform
                rms_value = np.sqrt(np.sum(np.square(waveform)) / waveform.size)

            noise = self.__handler_noise.gen_noise_real_pwr(out.size) if self.__add_noise else np.zeros_like(out)
            time = np.linspace(0, out.size, out.size, endpoint=False) / self._sampling_rate
            return WaveformSignal(
                time=time,
                signal=out+noise,
                fs=self._sampling_rate,
                rms=rms_value,
            )

    def generate_waveform_quant_fxp(self, time_points: list, time_duration: list,
                                    waveform_select: list, polarity_cathodic: list,
                                    bitwidth: int, bitfrac: int, signed: bool, do_opt: bool=False) -> WaveformSignal:
        """Generating the signal with waveforms for stimulation in quantized matter
        :param time_points:         List of time points for applying a stimulation waveform
        :param time_duration:       List of stimulation waveform duration
        :param waveform_select:     List of selected waveforms
        :param polarity_cathodic:   List for performing cathodic-first generation
        :param bitwidth:            Integer with total bitwidth
        :param bitfrac:             Integer with fraction bitwidth
        :param signed:              If quantized output should be signed integer
        :param do_opt:              Boolean for taking quarter signal (optimzed version for hardware implementation)
        :returns:                   Dataclass WaveformSignal with quantized signals ['time', 'signal', 'fs', 'rms']
        """
        assert check_keylist_elements_any(waveform_select, ['SINE_FULL', 'RECT_FULL', 'TRI_FULL']), "Only 'waveform_select' with ['SINE_FULL', 'RECT_FULL', 'TRI_FULL'] are allowed!"
        wvf_norm = self.generate_waveform(
            time_points=time_points,
            time_duration=time_duration,
            waveform_select=waveform_select,
            polarity_cathodic=polarity_cathodic
        )

        wvf_used = wvf_norm.signal / (wvf_norm.signal.max() - wvf_norm.signal.min()) + (0 if signed or do_opt else 0.5)
        wvf_used = np.array(wvf_used * (2**(bitwidth-bitfrac)), dtype=np.int32)
        if do_opt:
            wvf_used = wvf_used[:wvf_norm.signal.argmax() + 1]

        config_fxp = Config()
        config_fxp.rounding = "around"
        config_fxp.overflow = "saturate"
        config_fxp.underflow = "saturate"
        wvf_quant = Fxp(val=wvf_used, signed=signed, n_word=bitwidth, n_frac=bitfrac, config=config_fxp).get_val()
        return WaveformSignal(
            time=wvf_norm.time,
            signal=wvf_quant,
            fs=wvf_norm.fs,
            rms=wvf_norm.rms,
        )

    def generate_biphasic_waveform(self, anodic_wvf: str, anodic_duration: float,
                                   cathodic_wvf: str, cathodic_duration: float,
                                   intermediate_duration: float=0.0, do_cathodic_first: bool=False,
                                   do_charge_balancing: bool=False) -> dict:
        """Generating the waveform for stimulation
        Args:
            anodic_wvf:             String with waveform type for anodic phase
            anodic_duration:        Time window of the anodic phase
            cathodic_wvf:           String with waveform type for cathodic phase
            cathodic_duration:      Time window of the cathodic phase
            intermediate_duration:  Time window for the intermediate idle time during anodic and cathodic phase
            do_cathodic_first:      Starting with cathodic phase
            do_charge_balancing:    Performing a charge balancing on second phase (same area)
        Returns:
            Two numpy arrays (time, output_signal)
        """
        width = [anodic_duration, cathodic_duration] if not do_cathodic_first else [cathodic_duration, anodic_duration]
        mode = [anodic_wvf, cathodic_wvf] if not do_cathodic_first else [cathodic_wvf, anodic_wvf]
        poly = [False, True] if not do_cathodic_first else [True, False]
        waveforms = list()

        # --- Creating the waveforms
        for idx, (window, wvf_type, inverter) in enumerate(zip(width, mode, poly)):
            if idx == 1 and not intermediate_duration == 0.0:
                self._time_duration = intermediate_duration
                waveforms.append(self.__generate_zero())
            waveforms.append(self.__select_waveform_template(window, wvf_type, inverter))

        if do_charge_balancing:
            waveform = self.__get_charge_balancing_factor(waveforms) * waveforms[-1]
            waveforms[-1] = waveform

        # --- Creating the output signal
        out = np.concatenate([waveform for waveform in waveforms], axis=0)
        out = np.concatenate((out, np.zeros((1,))), axis=0)
        noise = self.__handler_noise.gen_noise_real_pwr(out.size) if self.__add_noise else np.zeros_like(out)
        time = np.linspace(0, out.size, out.size) / self._sampling_rate
        return {'t': time, 'y': out + noise}

    @staticmethod
    def build_random_timestamps(count: int, min_gap: float = 0.002, max_gap: float = 0.01) -> list:
        """Function for building random and sorted timestamps for generating waveforms
        :param count:       Number of timestamps to generate
        :param min_gap:     Minimum gap between timestamps [s]
        :param max_gap:     Maximum gap between timestamps [s]
        """
        values = []
        for _ in range(count):
            gap = np.random.uniform(min_gap, max_gap)
            if len(values):
                values.append(values[-1] + gap)
            else:
                values.append(gap)
        return values
