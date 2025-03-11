import numpy as np
from scipy import signal
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise, RecommendedSettingsNoise


class WaveformGenerator:
    __handler_noise: ProcessNoise

    def __init__(self, sampling_rate: float, add_noise: bool=False, settings_noise: SettingsNoise=RecommendedSettingsNoise):
        """Class for generating the transient stimulation signal
        :param sampling_rate:   Sampling rate of the signal
        :param add_noise:       Boolean for adding noise to output
        :param settings_noise:  Settings noise to add to output
        """
        self.__handler_noise = ProcessNoise(settings_noise, sampling_rate)
        self.__add_noise = add_noise
        self._sampling_rate = sampling_rate

        self.__func_dict = {'RECT': self.__generate_rectangular}
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
        self.__func_dict.update({'NOISE_ABS_RAND': self.__generate_sawtooth_negative})
        self.__func_dict.update({'ZERO': self.__generate_zero})

    @staticmethod
    def __switching_polarity(signal_in: np.ndarray, do_cathodic: bool) -> np.ndarray:
        """Switching the polarity for cathodic-first (True) or anodic-first (False) waveform"""
        return signal_in if not do_cathodic else (-1) * signal_in

    @staticmethod
    def __get_charge_balancing_factor(waveforms: list) -> float:
        """Getting the coefficient for area-related comparison for charge balancing the biphasic waveform"""
        if not len(waveforms) == 2 and not len(waveforms) == 3:
            print("It is not a biphasic waveform available - Please check!")
            return 1.0
        else:
            area_first = np.trapz(waveforms[0])
            area_second = np.trapz(waveforms[-1])
            return np.abs(area_first / area_second)

    @staticmethod
    def check_charge_balancing(signal: np.ndarray) -> float:
        """Checking if stimulation signal is charge balanced"""
        dq = np.trapz(signal)
        print(f"... waveform has an error of {dq:.6f}")
        return dq

    def __generate_zero(self, time_duration: float) -> np.ndarray:
        """Creating an output array with zero value"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.zeros((num_samples,), dtype=float)
        return out

    def __generate_rectangular(self, time_duration: float) -> np.ndarray:
        """Creating an output array with constant value"""
        return 1.0 + self.__generate_zero(time_duration)

    def __generate_linear_rising(self, time_duration: float) -> np.ndarray:
        """Creating an output array with linear positive slope"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.linspace(0.0, 1.0,  num_samples, dtype=float)
        return out

    def __generate_linear_falling(self, time_duration: float) -> np.ndarray:
        """Creating an output array with linear negative slope"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.linspace(1.0, 0.0,  num_samples, dtype=float)
        return out

    def __generate_sinusoidal_half(self, time_duration: float) -> np.ndarray:
        """Creating an output array with half sinusoidal waveform"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.sin(np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=False) / num_samples, dtype=float)
        return out

    def __generate_sinusoidal_half_inverse(self, time_duration: float) -> np.ndarray:
        """Creating an output array with half sinusoidal waveform in inverse manner"""
        num_samples = int(time_duration * self._sampling_rate)
        out = 1.0 - np.sin(np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=False) / num_samples, dtype=float)
        return out

    def __generate_sinusoidal_full(self, time_duration: float) -> np.ndarray:
        """Creating an output array with full sinusoidal waveform"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.sin(2 * np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=False) / num_samples, dtype=float)
        return out

    def __generate_triangle_half(self, time_duration: float) -> np.ndarray:
        """Creating an output array with half triangular waveform"""
        out0 = self.__generate_linear_rising(0.5 * time_duration)
        out1 = self.__generate_linear_falling(0.5 * time_duration)
        return np.concatenate((out0, out1), axis=0)

    def __generate_triangle_full(self, time_duration: float) -> np.ndarray:
        """Creating an output array with full triangular waveform"""
        out0 = self.__generate_linear_rising(0.25 * time_duration)
        out1 = self.__generate_linear_falling(0.25 * time_duration)
        return np.concatenate((out0, out1, -out0, -out1), axis=0)

    def __generate_sawtooth_positive(self, time_duration: float) -> np.ndarray:
        """Creating an output array with linear positive sawtooth"""
        return 2 * self.__generate_linear_rising(time_duration) - 1.0

    def __generate_sawtooth_negative(self, time_duration: float) -> np.ndarray:
        """Creating an output array with linear negative sawtooth"""
        return 2 * self.__generate_linear_falling(time_duration) - 1.0

    def __generate_gaussian(self, time_duration: float) -> np.ndarray:
        """Creating an output array with gaussian pulse"""
        time = 2 * self.__generate_sawtooth_positive(time_duration)
        out = signal.gausspulse(time, 2.72, retenv=True)
        return out[1]

    def get_dictionary_classes(self, do_print: bool=False) -> list:
        """Getting a list with class names"""
        out_list = [val for val in self.__func_dict.keys()]
        if do_print:
            print("\nGetting information about signal types")
            print("\n====================================================")
            for idx, type_id in enumerate(out_list):
                print(f"Class {idx:02d} = {type_id}")
            print("====================================================")
        return out_list

    def __select_waveform_template(self, time_duration: float, sel_wfg: int, do_cathodic: bool=False) -> np.ndarray:
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
        class_name = self.get_dictionary_classes()

        if class_name[sel_wfg] in self.__func_dict.keys():
            signal = self.__func_dict[class_name[sel_wfg]](time_duration)
        else:
            print("Signal not available")
            signal = self.__generate_zero(time_duration)
        waveform = self.__switching_polarity(signal, do_cathodic)
        return waveform

    def generate_waveform(self, time_points: list, time_duration: list,
                          waveform_select: list, polarity_cathodic: list) -> dict:
        """Generating the signal with waveforms for stimulation
        Args:
            time_points:        List of time points for applying a stimulation waveform
            time_duration:      List of stimulation waveform duration
            waveform_select:    List of selected waveforms
            polarity_cathodic:  List for performing cathodic-first generation
        Returns:
            List with three numpy arrays (time, output_signal, true rms value)
        """
        if not len(time_points) == len(waveform_select) == len(time_duration):
            raise RuntimeError("Please check input! --> Length is not equal")
        else:
            # Generate dummy
            out = self.__generate_zero(2 * time_points[-1] + time_duration[-1])
            time = np.linspace(0, out.size, out.size, endpoint=False) / self._sampling_rate
            rms_value = 0.0

            # Create waveform
            for idx, time_sec in enumerate(time_points):
                do_polarity = polarity_cathodic[idx] if not len(polarity_cathodic) == 0 else False
                time_xpos = int(time_sec * self._sampling_rate)

                waveform = self.__select_waveform_template(time_duration[idx], waveform_select[idx], do_polarity)
                out[time_xpos:time_xpos+waveform.size] = waveform
                noise = self.__handler_noise.gen_noise_real_pwr(out.size) if self.__add_noise else np.zeros_like(out)
                rms_value = np.sqrt(np.sum(np.square(waveform)) / waveform.size)
            return {'time': time, 'sig': out + noise, 'rms': rms_value}

    def generate_biphasic_waveform(self, anodic_mode: int, anodic_duration: float,
                                   cathodic_mode: int, cathodic_duration: float,
                                   intermediate_duration: float=0.0, do_cathodic_first: bool=False,
                                   do_charge_balancing: bool=False) -> dict:
        """Generating the waveform for stimulation
        Args:
            anodic_mode:            Mode of the anodic phase
            anodic_duration:        Time window of the anodic phase
            cathodic_mode:          Mode of cathodic phase
            cathodic_duration:      Time window of the cathodic phase
            intermediate_duration:  Time window for the intermediate idle time during anodic and cathodic phase
            do_cathodic_first:      Starting with cathodic phase
            do_charge_balancing:    Performing a charge balancing on second phase (same area)
        Returns:
            Two numpy arrays (time, output_signal)
        """
        width = [anodic_duration, cathodic_duration] if not do_cathodic_first else [cathodic_duration, anodic_duration]
        mode = [anodic_mode, cathodic_mode] if not do_cathodic_first else [cathodic_mode, anodic_mode]
        poly = [False, True] if not do_cathodic_first else [True, False]
        waveforms = list()

        # --- Creating the waveforms
        for idx, window in enumerate(width):
            if idx == 1 and not intermediate_duration == 0.0:
                waveforms.append(self.__generate_zero(intermediate_duration))
            waveforms.append(self.__select_waveform_template(window, mode[idx], poly[idx]))

        if do_charge_balancing:
            waveform = self.__get_charge_balancing_factor(waveforms) * waveforms[-1]
            waveforms[-1] = waveform

        # --- Creating the output signal
        out = np.concatenate([waveform for waveform in waveforms], axis=0)
        noise = self.__handler_noise.gen_noise_real_pwr(out.size) if self.__add_noise else np.zeros_like(out)
        time = np.linspace(0, out.size, out.size) / self._sampling_rate
        return {'t': time, 'y': out + noise}
