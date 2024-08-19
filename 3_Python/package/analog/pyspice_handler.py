import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def resistor(value: float) -> Circuit:
    """"""
    circuit = Circuit("Resistive Load")
    circuit.R(1, 'input', 'output', value)
    circuit.V('cm', 'output', circuit.gnd, 0.0)
    return circuit


def resistive_diode(r0: float, Uth=0.7, IS0=4e-12, N=2) -> Circuit:
    """"""
    circuit = Circuit("Resistive Diode")
    circuit.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12,)
    circuit.R(1, 'input', 'middle', r0)
    circuit.Diode(0, 'middle', 'output', model='MyDiode')
    circuit.V('cm', 'output', circuit.gnd, 0.0)
    return circuit


def resistive_diode_antiparallel(r0: float, Uth=0.7, IS0=4e-12, N=2) -> Circuit:
    """"""
    circuit = Circuit("Resistive Diode (Antiparallel)")
    circuit.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12,)
    circuit.R(1, 'input', 'middle', r0)
    circuit.Diode(0, 'middle', 'output', model='MyDiode')
    circuit.Diode(1, 'output', 'middle', model='MyDiode')
    circuit.V('cm', 'output', circuit.gnd, 0.0)
    return circuit


def simple_randles_model(R_tis=10e3, R_far=100e6, C_dl=10e-9) -> Circuit:
    """"""
    circuit = Circuit("Simple Randles Model")
    circuit.R(1, 'input', 'middle', R_tis)
    circuit.R(2, 'middle', 'output', R_far)
    circuit.C(1, 'middle', 'output', C_dl)
    circuit.V('cm', 'output', circuit.gnd, 0.0)
    return circuit


def voltage_divider(r_0: float, r_1: float, r_load=10e9, c_load=0.0) -> Circuit:
    """"""
    circuit = Circuit("Voltage Divider with Load")
    circuit.R(1, 'input', 'output', r_0)
    circuit.R(2, 'output', circuit.gnd, r_1)
    circuit.R(3, 'output', circuit.gnd, r_load)
    if not c_load == 0.0:
        circuit.C(0, 'output', circuit.gnd, c_load)
    return circuit


def _raise_voltage_violation(du: np.ndarray | float, range_volt: list) -> None:
    """Checking differential voltage input for violation of voltage range for given branch"""
    violation_dwn = np.count_nonzero(du < range_volt[0], axis=0)
    violation_up = np.count_nonzero(du > range_volt[1], axis=0)

    if violation_up or violation_dwn:
        warn("Warning: Voltage Range Violation! - Results are not confirmed!", DeprecationWarning)
        addon_dwn = '' if not violation_dwn else '(Downer limit)'
        addon_up = '' if not violation_up else ' (Upper limit)'
        print(f"Warning: Voltage Range Violation! {addon_dwn}{addon_up}")


class _ArbWFG(NgSpiceShared):
    def __init__(self, waveform: np.ndarray, f_samp: float, amplitude=1.0, **kwargs) -> None:
        """Private Class for Enabling Transient Simulation with Custom-defined Arbitrary Waveforms
        Args:
            waveform:   Transient input signal
            f_samp:     Sampling rate
            amplitude:  Peak amplitude of signal [Default: 1.0]
        Returns:
            None
        """
        super().__init__(**kwargs)
        self._amplitude = amplitude
        self._waveform = waveform
        self._f_samp_wfg = f_samp

    def get_vsrc_data(self, voltage, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        index = int(time * self._f_samp_wfg) % len(self._waveform)
        voltage[0] = self._amplitude * self._waveform[index]
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
        index = int(time * self._f_samp_wfg) % len(self._waveform)
        current[0] = self._amplitude * self._waveform[index]
        return 0


class PySpice_Handler:
    _circuit:            Circuit
    _is_input_voltage:   bool
    __results:           None

    def __init__(self, temperature=300.0, input_voltage=True) -> None:
        """
        Easy Tutorial for Using PySpice: https://github.com/benedictjones/engineeringthings-pyspice (Git),
        https://www.youtube.com/watch?v=62BOYx1UCfs&list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh (YouTube)
        Args:
            temperature:    Given temperature for simulation in [K] [Default: 300.0 K]
            input_voltage:  Defining if input is a voltage (True) or current (False)
        Returns:
            None
        """
        self._is_input_voltage = input_voltage
        self._used_temp = temperature
        self.__plot_color = 'krbg'
        self._circuit = Circuit("Test")

    @property
    def __calc_temp_in_celsius(self) -> float:
        """Translating the temperature value from Kelvin [K] to Grad Celsius [°C]"""
        return self._used_temp - 273.15

    def set_src_mode(self, do_voltage: bool) -> None:
        """Setting the Source Mode of Input Source [0: Current, 1: Voltage]"""
        self._is_input_voltage = do_voltage

    def load_circuit_model(self, circuit: Circuit) -> None:
        """Loading an external circuit SPICE model"""
        self._circuit = circuit

    def print_spice_circuit(self) -> None:
        """Printing the circuit in SPICE format"""
        print(self._circuit)

    def do_dc_simulation(self, value: float, do_print_results=True, initial_value=0.0) -> dict:
        """Performing the DC or Operating Point Simulation
        Args:
            value:              Specified value of the input voltage or current source
            do_print_results:   Printing the node voltages
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            Dictionary with node voltages
        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, value)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', value)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        analysis = simulator.operating_point()
        self.__results = analysis

        results = dict()
        for node in analysis.nodes.values():
            results.update({str(node): float(node)})
            if do_print_results:
                unit = 'V'
                print(f'Node {str(node)}: {float(node):4.3f} {unit}')
        return results

    def do_dc_sweep_simulation(self, start_dc: float, stop_dc: float, step_dc: float,
                               initial_value=0.0):
        """Performing the DC or Operating Point Simulation
        Args:
            start_dc:   Starting point of DC Sweep
            stop_dc:    End point of DC Sweep
            step_dc:    Step size of DC Sweep
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, 0)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', 0)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)

        if self._is_input_voltage:
            self.__results = simulator.dc(Vinput=slice(start_dc, stop_dc, step_dc))
        else:
            self.__results = simulator.dc(Iinput=slice(start_dc, stop_dc, step_dc))

        return self.__results

    def do_ac_simulation(self, start_freq: float, stop_freq: float, num_points: int,
                         amplitude=1.0, initial_value=0.0):
        """Performing the DC or Operating Point Simulation
        Args:
            start_freq:     Frequency value for starting point
            stop_freq:      Frequency value for end point
            num_points:     Number of repetions
            amplitude:      Amplitude of input signal [Default: 1.0]
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        if self._is_input_voltage:
            self._circuit.SinusoidalVoltageSource('input', 'input', self._circuit.gnd, amplitude)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.SinusoidalCurrentSource('input', self._circuit.gnd, 'input', amplitude)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        self.__results = simulator.ac(start_frequency=start_freq, stop_frequency=stop_freq,
                                      number_of_points=num_points, variation='dec')
        return self.__results

    def do_transient_pulse_simulation(self, neg_value: float, pos_value: float,
                                      pulse_width: float, pulse_period: float,
                                      t_sim: float, f_samp: float, initial_value=0.0):
        """Performing the Transient Simulation with Pulse Signal
        Args:
            neg_value:      Pos. peak value of the pulse signal [V or A]
            pos_value:      Neg. peak value of the pulse signal [V or A]
            pulse_width:    Pulse width [s]
            pulse_period:   Period of the pulses [s]
            t_sim:          Total simulation time [s]
            f_samp:         Sampling frequency [Hz]
            initial_value:  Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        if self._is_input_voltage:
            self._circuit.PulseVoltageSource('input', 'input', self._circuit.gnd,
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.PulseCurrentSource('input', self._circuit.gnd, 'input',
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        self.__results = simulator.transient(step_time=1/f_samp, end_time=t_sim)
        return self.__results

    def do_transient_sinusoidal_simulation(self, amp: float, freq: float,
                                           t_sim: float, f_samp: float,
                                           t_dly=0.0, offset=0.0, initial_value=0.0):
        """Performing the Transient Simulation with Sinusoidal Signal
        Args:
            amp:            Amplitude of sinusoidal waveform [V or A]
            freq:           Frequency of sinusoidal waveform [Hz]
            t_sim:          Total simulation time [s]
            f_samp:         Sampling frequency [Hz]
            t_dly:          Applied time delay to signal [s] [Default: 0.0]
            offset:         Applied offset on signal [V or A] [Default: 0.0]
            initial_value:  Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        if self._is_input_voltage:
            self._circuit.SinusoidalVoltageSource(
                'input', 'input', self._circuit.gnd,
                amplitude=amp, delay=t_dly, offset=offset, frequency=freq
            )
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.SinusoidalCurrentSource(
                'input', self._circuit.gnd, 'input',
                amplitude=amp, delay=t_dly, offset=offset, frequency=freq
            )
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        self.__results = simulator.transient(step_time=1/f_samp, end_time=t_sim)
        return self.__results

    def do_transient_arbitary_simulation(self, signal: np.ndarray, t_end: float, f_samp: float, initial_value=0.0):
        """Performing the Transient Simulation with Pulse Signal
        Args:
            signal:     Numpy array with transient custom-made signal
            t_end:      Total simulation time [s]
            f_samp:     Sampling frequency [Hz]
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, 'dc 0 external')
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', 'dc 0 external')
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        data = _ArbWFG(signal, f_samp, amplitude=1e0, send_data=False)
        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius,
                                            simulator='ngspice-shared', ngspice_shared=data)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        self.__results = simulator.transient(step_time=1/f_samp, end_time=t_end)
        return self.__results

    def create_dummy_signal(self, t_sim: float, f_samp: float, offset=0.0) -> [np.ndarray, np.ndarray]:
        """Creating a dummy function for transient simulation
        Args:
            t_sim:  Simulation time [s]
            f_samp: Sampling frequency [Hz]
            offset: Offset on signal [Default: 0.0]
        Returns:
            Two numpy arrays with time vector and signal vector
        """
        freq_used = [100, 300, 500]
        amp0_used = [1.0, 0.25, 0.66]

        time0 = np.linspace(0, t_sim, int(t_sim * f_samp), endpoint=True)

        sig_out = np.zeros(time0.shape) + offset
        for freq, amp in zip(freq_used, amp0_used):
            sig_out += amp * np.sin(2 * np.pi * freq * time0)
        return time0, sig_out

    def get_results(self, mode: int) -> dict:
        """Getting the results from SPICE analysis
        Args:
            mode:   Selection mode for getting results (0 = Operating Point, 1 = DC Sweep, 2 = AC Sweep, 3 = Transient)
        Returns:
              Dictionary with entries "v_in", "v_out", "i_in", "time", "freq"
        """
        output_dict = dict()
        cur_in_selection = True if 'viinput_plus' in self.__results.branches.keys() else False
        cur_in_key = 'vvinput_minus' if not cur_in_selection else 'viinput_plus'
        match mode:
            case 0:
                # DC Operating Point Analysis
                output_dict.update({"v_in": np.array(self.__results.input)})
                output_dict.update({"i_in": np.array(self.__results.branches[cur_in_key])})
            case 1:
                # DC Sweep Analysis
                output_dict.update({"v_in": np.array(self.__results.input)})
                output_dict.update({"i_in": np.array(self.__results.branches[cur_in_key])})
            case 2:
                # AC Sweep Analysis
                output_dict.update({"freq": np.array(self.__results.frequency)})
                output_dict.update({"v_in": np.array(self.__results.input)})
                output_dict.update({"i_in": np.array(self.__results.branches[cur_in_key])})
                output_dict.update({"v_out": np.array(self.__results.output)})
            case 3:
                # Transient Simulation (Pulse, Sinusoidal or Arbitrary)
                output_dict.update({"time": np.array(self.__results.time)})
                output_dict.update({"v_in": np.array(self.__results.input)})
                output_dict.update({"i_in": np.array(self.__results.branches[cur_in_key])})
                output_dict.update({"v_out": np.array(self.__results.output)})
        return output_dict

    def plot_iv_curve(self, do_log=False, path2save='') -> None:
        """Plotting the I-V relationship/curve of investigated circuit (taking v_in and i_in)
        Args:
            do_log:     Do a logarithmic plotting on y-axis
            path2save:  Optional string for plotting [Default: '' for non-plotting]
        Returns:
          None
        """
        # --- Getting data
        results = self.get_results(0)
        u_in = results["v_in"]
        i_out = results["i_in"]

        # --- Plotting
        plt.figure()
        if not do_log:
            plt.plot(u_in, i_out, 'k', linewidth=1, marker='.')
        else:
            plt.semilogy(u_in, np.abs(i_out), 'k', linewidth=1, marker='.')
        plt.xlabel(r'Voltage $U$ / V')
        plt.ylabel(r'Current $I$ / A')
        plt.xlim([u_in[0], u_in[-1]])

        plt.tight_layout()
        plt.grid()
        if path2save:
            plt.savefig('test.svg', format='svg')
        plt.show(block=True)

    def plot_bodeplot(self, mode=0, path2save='') -> None:
        """Plotting the Bode Diagram (mode == 0) or Impedance Plot (mode == 1) of investigated circuit
        Args:
            mode:       Mode selection (0 = Bode diagram, 1 = Impedance plot)
            path2save:  Optional string for plotting [Default: '' for non-plotting]
        Returns:
          None
        """
        # --- Getting data
        results = self.get_results(2)
        freq = results["freq"]
        transfer_function = results["v_out"] / results["v_in"] if mode == 0 else results["v_in"] / results["i_in"]

        # --- Plotting
        fig, axs = plt.subplots(2, 1, sharex="all")
        axs[0].semilogx(freq, 20 * np.log10(np.abs(transfer_function)), 'k', linewidth=1, marker='.', label="Gain")
        axs[0].set_ylabel(r"Gain $v_U$ / dB")
        axs[0].set_xlim([freq[0], freq[-1]])

        axs[1].semilogx(freq, np.angle(transfer_function, deg=True), 'r', linewidth=1, marker='.', label="Phase")
        axs[1].set_ylabel(r"Phase $\alpha$ / °")
        axs[1].set_xlabel(r'Frequency $f$ / dB')
        axs[1].set_xlim([freq[0], freq[-1]])

        for ax in axs:
            ax.set_xlim([freq[0], freq[-1]])
            ax.grid(which='both', linestyle='--')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        if path2save:
            plt.savefig('test.svg', format='svg')
        plt.show(block=True)

    def plot_transient(self, path2save='') -> None:
        """Plotting the results of Transient Simulation of investigated circuit
        Args:
            path2save:  Optional string for plotting [Default: '' for non-plotting]
        Returns:
          None
        """
        # --- Getting data
        results = self.get_results(3)
        time = results["time"]
        v_sig = [results["v_in"], results["v_out"]]
        v_label = ["Input", "Output"]
        i_in = results["i_in"]

        # --- Plotting
        fig, ax1 = plt.subplots()
        idx = 0
        for sig, label in zip(v_sig, v_label):
            if idx == 0:
                lns = ax1.plot(time, sig, linewidth=1, label=label)
            else:
                lns += ax1.plot(time, sig, linewidth=1, label=label)
            idx += 1

        ax1.set_ylabel(r"Voltage $U_x$ / V")
        ax1.set_xlabel(r'Time $t$ / s')

        ax2 = ax1.twinx()
        lns += ax2.plot(time, i_in, 'r--', linewidth=1, label='Current')
        ax2.set_ylabel(r"Current $I$ / A")
        ax2.set_xlim([time[0], time[-1]])

        # added these three lines
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0)

        plt.tight_layout()
        plt.grid()
        if path2save:
            plt.savefig('test.svg', format='svg')
        plt.show(block=True)
