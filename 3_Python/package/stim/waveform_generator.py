import numpy as np
from scipy import signal


class WaveformGenerator:
    """Class for generating the transient stimulation signal"""
    def __init__(self, sampling_rate: float):
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
        self.__func_dict.update({'NOISE_RAND': self.__generate_random_noise})
        self.__func_dict.update({'NOISE_ABS_RAND': self.__generate_sawtooth_negative})
        self.__func_dict.update({'ZERO': self.__generate_zero})

    def __switching_polarity(self, signal_in: np.ndarray, do_cathodic: bool) -> np.ndarray:
        """Switching the polarity for cathodic-first (True) or anodic-first (False) waveform"""
        return signal_in if not do_cathodic else (-1) * signal_in

    def __get_charge_balancing_factor(self, waveforms: list) -> float:
        """Getting the coefficient for area-related comparison for charge balancing the biphasic waveform"""
        if not len(waveforms) == 2 and not len(waveforms) == 3:
            print("It is not a biphasic waveform available - Please check!")
            return 1.0
        else:
            area_first = np.trapz(waveforms[0])
            area_second = np.trapz(waveforms[-1])
            return np.abs(area_first / area_second)

    def check_charge_balancing(self, signal: np.ndarray) -> float:
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
        out = np.sin(np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=True) / num_samples, dtype=float)
        return out

    def __generate_sinusoidal_half_inverse(self, time_duration: float) -> np.ndarray:
        """Creating an output array with half sinusoidal waveform in inverse manner"""
        num_samples = int(time_duration * self._sampling_rate)
        out = 1.0 - np.sin(np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=True) / num_samples, dtype=float)
        return out

    def __generate_sinusoidal_full(self, time_duration: float) -> np.ndarray:
        """Creating an output array with full sinusoidal waveform"""
        num_samples = int(time_duration * self._sampling_rate)
        out = np.sin(2 * np.pi * np.linspace(0.0, num_samples, num_samples, endpoint=True) / num_samples, dtype=float)
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

    def __generate_random_noise(self, time_duration: float) -> np.ndarray:
        """Creating an output array with random noise (gaussian distribution)"""
        num_samples = int(time_duration * self._sampling_rate)
        return 2 * np.random.random(num_samples) - 1.0

    def __generate_random_noise_abs(self, time_duration: float) -> np.ndarray:
        """Creating an output array with random absolute noise (gaussian distribution)"""
        return np.abs(self.__generate_random_noise(time_duration))

    def get_dictionary_classes(self) -> list:
        """Getting a list with class names"""
        out_list = list()
        for val in self.__func_dict.keys():
            out_list.append(val)
        return out_list

    def __select_waveform_template(self, time_duration: float, sel_wfg: int, do_cathodic=False) -> np.ndarray:
        """Selection for generating a waveform template
        Args:
            time_duration:  Time window for the waveform
            sel_wfg:        Selected waveform type [0: rect., 1: linear-rising, 2: linear-falling, 3: half-sinusoidal,
                            4: half-sinusoidal (inverse), 5: full-sinusoidal, 6: half-triangular, 7: full-triangular,
                            8: positive sawtooth, 9: negative sawtooth, 10: gaussian, 11: random absolute noise]
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

        return self.__switching_polarity(signal, do_cathodic)

    def generate_waveform(self, time_points: list, time_duration: list,
                          waveform_select: list, polarity_cathodic: list) -> list:
        """Generating the signal with waveforms for stimulation
        Args:
            time_points:        List of time points for applying a stimulation waveform
            time_duration:      List of stimulation waveform duration
            waveform_select:    List of selected waveforms
            polarity_cathodic:  List for performing cathodic-first generation
        Returns:
            Two numpy arrays (time, output_signal)
        """
        if not len(time_points) == len(waveform_select) == len(time_duration):
            print("Please check input! --> Length is not equal")
            return list()
        else:
            # Generate dummy
            out = self.__generate_zero(2 * time_points[-1] + time_duration[-1])
            time = np.linspace(0, out.size, out.size, endpoint=True) / self._sampling_rate

            # Create waveform
            for idx, time_sec in enumerate(time_points):
                do_polarity = polarity_cathodic[idx] if not len(polarity_cathodic) == 0 else False
                time_xpos = int(time_sec * self._sampling_rate)

                waveform = self.__select_waveform_template(time_duration[idx], waveform_select[idx], do_polarity)
                out[time_xpos:time_xpos+waveform.size] = waveform

            return [time, out]

    def generate_biphasic_waveform(self, anodic_mode: int, anodic_duration: float,
                                   cathodic_mode: int, cathodic_duration: float,
                                   intermediate_duration=0.0, do_cathodic_first=False,
                                   do_charge_balancing=False) -> list:
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
            k = self.__get_charge_balancing_factor(waveforms)
            waveform = k * waveforms[-1]
            waveforms[-1] = waveform

        # --- Creating the output signal
        out = np.concatenate([waveform for waveform in waveforms], axis=0)
        time = np.linspace(0, out.size, out.size) / self._sampling_rate
        return [time, out]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    num_elements = 13
    time_points = [0.2 + 0.4 * idx for idx in range(num_elements)]
    time_duration = [0.1 for idx in range(num_elements)]
    time_wfg = [idx for idx in range(num_elements)]
    polarity_cathodic = [False for idx in range(num_elements)]
    # polarity_cathodic = [idx % 3 == 0 for idx in range(num_elements)]

    wfg_generator = WaveformGenerator(50e3)
    t0, signal0 = wfg_generator.generate_waveform(time_points, time_duration, time_wfg, polarity_cathodic)
    t1, signal1 = wfg_generator.generate_biphasic_waveform(0, 0.1, 0, 0.2, 0.05, True, True)
    wfg_generator.check_charge_balancing(signal1)

    # --- Plotting: All waveforms
    plt.figure()
    plt.plot(t0, signal0, 'k')
    plt.xlabel("Time t / s")
    plt.ylabel("Signal y(t)")
    plt.grid()
    plt.tight_layout()

    # --- Plotting: Biphasic waveform
    plt.figure()
    plt.plot(t1, signal1, 'k')
    plt.xlabel("Time t / s")
    plt.ylabel("Signal y(t)")
    plt.grid()
    plt.tight_layout()
    plt.show()
