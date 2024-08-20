import numpy as np
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


def resistor(value: float) -> Circuit:
    """"""
    circuit0 = Circuit("Resistive Load")
    circuit0.R(1, 'input', 'output', value)
    circuit0.V('cm', 'output', circuit0.gnd, 0.0)
    return circuit0


def resistive_diode(r0: float, Uth=0.7, IS0=4e-12, N=2) -> Circuit:
    """"""
    circuit0 = Circuit("Resistive Diode")
    circuit0.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12, )
    circuit0.R(1, 'input', 'middle', r0)
    circuit0.Diode(0, 'middle', 'output', model='MyDiode')
    circuit0.V('cm', 'output', circuit0.gnd, 0.0)
    return circuit0


def resistive_diode_antiparallel(r0: float, Uth=0.7, IS0=4e-12, N=2) -> Circuit:
    """"""
    circuit0 = Circuit("Resistive Diode (Antiparallel)")
    circuit0.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12, )
    circuit0.R(1, 'input', 'middle', r0)
    circuit0.Diode(0, 'middle', 'output', model='MyDiode')
    circuit0.Diode(1, 'output', 'middle', model='MyDiode')
    circuit0.V('cm', 'output', circuit0.gnd, 0.0)
    return circuit


def simple_randles_model(R_tis=10e3, R_far=100e6, C_dl=10e-9) -> Circuit:
    """"""
    circuit0 = Circuit("Simple Randles Model")
    circuit0.R(1, 'input', 'middle', R_tis)
    circuit0.R(2, 'middle', 'output', R_far)
    circuit0.C(1, 'middle', 'output', C_dl)
    circuit0.V('cm', 'output', circuit0.gnd, 0.0)
    return circuit0


def voltage_divider(r_0: float, r_1: float, r_load=10e9, c_load=0.0) -> Circuit:
    """"""
    circuit0 = Circuit("Voltage Divider with Load")
    circuit0.R(1, 'input', 'output', r_0)
    circuit0.R(2, 'output', circuit0.gnd, r_1)
    circuit0.R(3, 'output', circuit0.gnd, r_load)
    if not c_load == 0.0:
        circuit0.C(0, 'output', circuit0.gnd, c_load)
    return circuit0


class _ArbWFG(NgSpiceShared):
    def __init__(self, waveform: np.ndarray, f_samp: float, **kwargs) -> None:
        """Private Class for Enabling Transient Simulation with Custom-defined Arbitrary Waveforms
        Args:
            waveform:   Transient input signal
            f_samp:     Sampling rate
        Returns:
            None
        """
        super().__init__(**kwargs)

        self.waveform = waveform
        self.f_samp_wfg = f_samp

    def get_vsrc_data(self, voltage, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        index = int(time * self.f_samp_wfg) % len(self.waveform)
        voltage[0] = self.waveform[index]
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
        index = int(time * self.f_samp_wfg) % len(self.waveform)
        current[0] = self.waveform[index]
        return 0


class PySpice_Handler:
    _circuit: Circuit
    _is_input_voltage: bool
    __results: dict

    def __init__(self, temperature=300.0, input_voltage=True) -> None:
        """Rewritten API for using PySPICE in simulation (Git Tutorial: https://github.com/benedictjones/engineeringthings-pyspice, YouTube: https://www.youtube.com/watch?v=62BOYx1UCfs&list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh)
        Args:
            temperature:    Given temperature for simulation in [K] [Default: 300.0 K]
            input_voltage:  Defining if input is a voltage (True) or current (False)
        Returns:
            None
        """
        self.__results = dict()
        self._is_input_voltage = input_voltage
        self._used_temp = temperature
        self.__plot_color = 'krbg'
        self._circuit = Circuit("Test")
        self._run_ite = 0
        self._arbitrary_signal_ng_spice_instance = None

    @property
    def __calc_temp_in_celsius(self) -> float:
        """Translating the temperature value from Kelvin [K] to Grad Celsius [°C]"""
        return self._used_temp - 273.15

    def set_src_mode(self, do_voltage: bool) -> None:
        """Setting the Source Mode of Input Source [0: Current, 1: Voltage]"""
        self._is_input_voltage = do_voltage

    def load_circuit_model(self, circuit_new: Circuit) -> None:
        """Loading an external circuit SPICE model"""
        self._circuit = circuit_new.clone()

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

        results = dict()
        for node in analysis.nodes.values():
            results.update({str(node): float(node)})
            if do_print_results:
                unit = 'V'
                print(f'Node {str(node)}: {float(node):4.3f} {unit}')

        del self.__results
        self.__results = self.get_results(0, analysis)
        return self.__results

    def do_dc_sweep_simulation(self, start_dc: float, stop_dc: float, step_dc: float,
                               initial_value=0.0) -> dict:
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
            results = simulator.dc(Vinput=slice(start_dc, stop_dc, step_dc))
        else:
            results = simulator.dc(Iinput=slice(start_dc, stop_dc, step_dc))

        del self.__results
        self.__results = self.get_results(1, results)
        return self.__results

    def do_ac_simulation(self, start_freq: float, stop_freq: float, num_points: int,
                         amplitude=1.0, initial_value=0.0) -> dict:
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
        results = simulator.ac(start_frequency=start_freq, stop_frequency=stop_freq,
                               number_of_points=num_points, variation='dec')

        del self.__results
        self.__results = self.get_results(2, results)
        return self.__results

    def do_transient_pulse_simulation(self, neg_value: float, pos_value: float,
                                      pulse_width: float, pulse_period: float,
                                      t_sim: float, f_samp: float, initial_value=0.0) -> dict:
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
                                             pulse_width=pulse_width, period=pulse_period,
                                             rise_time=1 / f_samp, fall_time=1 / f_samp)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.PulseCurrentSource('input', self._circuit.gnd, 'input',
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period,
                                             rise_time=1 / f_samp, fall_time=1 / f_samp)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.transient(step_time=1 / f_samp, end_time=t_sim)

        del self.__results
        self.__results = self.get_results(3, results)
        return self.__results

    def do_transient_sinusoidal_simulation(self, amp: float, freq: float,
                                           t_sim: float, f_samp: float,
                                           t_dly=0.0, offset=0.0, initial_value=0.0) -> dict:
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
        results = simulator.transient(step_time=1 / f_samp, end_time=t_sim)

        del self.__results
        self.__results = self.get_results(3, results)
        return self.__results

    def do_transient_arbitrary_simulation(self, signal: np.ndarray, t_end: float, f_samp: float,
                                          initial_value=0.0) -> dict:
        """Performing the Transient Simulation with Arbitrary Signal Waveform
        Args:
            signal:         Numpy array with transient custom-made signal
            t_end:          Total simulation time [s]
            f_samp:         Sampling frequency [Hz]
            initial_value:  Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        # TODO: Current controlled simulation does not work (I don't know why)

        # --- Definition of energy source
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, 'dc 0 external')
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', 'dc 0 external')
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        # --- Generating instance for using arbitrary waveforms in SPICE transient simulation
        if self._arbitrary_signal_ng_spice_instance is None:
            self._arbitrary_signal_ng_spice_instance = _ArbWFG(signal, f_samp, send_data=False)
        else:
            self._arbitrary_signal_ng_spice_instance.waveform = signal
            self._arbitrary_signal_ng_spice_instance.f_samp_wfg = f_samp
        data = self._arbitrary_signal_ng_spice_instance

        # --- Prepare and Run simulation
        simulator = self._circuit.simulator(temperature=self.__calc_temp_in_celsius,
                                            nominal_temperature=self.__calc_temp_in_celsius,
                                            simulator='ngspice-shared', ngspice_shared=data)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.transient(step_time=1 / f_samp, end_time=t_end)

        # --- Process results
        del self.__results
        self.__results = self.get_results(3, results)
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

    def get_results(self, mode: int, data) -> dict:
        """Getting the results from SPICE analysis
        Args:
            mode:   Selection mode for getting results (0 = Operating Point, 1 = DC Sweep, 2 = AC Sweep, 3 = Transient)
        Returns:
              Dictionary with entries "v_in", "v_out", "i_in", "time", "freq"
        """
        output_dict = dict()
        cur_in_selection = True if 'viinput_plus' in data.branches.keys() else False
        cur_in_key = 'vvinput_minus' if not cur_in_selection else 'viinput_plus'
        match mode:
            case 0:
                # DC Operating Point Analysis
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key])})
            case 1:
                # DC Sweep Analysis
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key])})
            case 2:
                # AC Sweep Analysis
                output_dict.update({"freq": np.array(data.frequency)})
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key])})
                output_dict.update({"v_out": np.array(data.output)})
            case 3:
                # Transient Simulation (Pulse, Sinusoidal or Arbitrary)
                output_dict.update({"time": np.array(data.time)})
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key])})
                output_dict.update({"v_out": np.array(data.output)})
        return output_dict

    def plot_iv_curve(self, do_log=False, path2save='', block_plots=True) -> None:
        """Plotting the I-V relationship/curve of investigated circuit (taking v_in and i_in)
        Args:
            do_log:     Do a logarithmic plotting on y-axis
            path2save:  Optional string for plotting [Default: '' for non-plotting]
            block_plots:    Blocking plots for showing [Default: True]
        Returns:
          None
        """
        # --- Getting data
        results = self.__results
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
        plt.show(block=block_plots)

    def plot_bodeplot(self, mode=0, path2save='', block_plots=True) -> None:
        """Plotting the Bode Diagram (mode == 0) or Impedance Plot (mode == 1) of investigated circuit
        Args:
            mode:           Mode selection (0 = Bode diagram, 1 = Impedance plot)
            path2save:      Optional string for plotting [Default: '' for non-plotting]
            block_plots:    Blocking plots for showing [Default: True]
        Returns:
          None
        """
        # --- Getting data
        results = self.__results
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
        plt.show(block=block_plots)

    def plot_transient(self, path2save='', block_plots=True) -> None:
        """Plotting the results of Transient Simulation of investigated circuit
        Args:
            path2save:  Optional string for plotting [Default: '' for non-plotting]
            block_plots:    Blocking plots for showing [Default: True]
        Returns:
          None
        """
        # --- Getting data
        results = self.__results
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
        plt.show(block=block_plots)


if __name__ == "__main__":
    # --- Settings
    run_mode = [0, 1, 2, 3, 4, 5, 5]
    do_voltage = True
    fs = 100e3
    t_sim = 20e-3

    circuit = voltage_divider(10e3, 10e3, c_load=10e-9)
    # --- Definition of Sim mode
    pyspice = PySpice_Handler(input_voltage=do_voltage)
    for mode in run_mode:
        pyspice.load_circuit_model(circuit)
        if mode == 0:
            pyspice.do_dc_simulation(1.0)
        elif mode == 1:
            pyspice.do_dc_sweep_simulation(-2.0, 5.0, 1e-3)
            pyspice.plot_iv_curve(do_log=False, block_plots=False)
        elif mode == 2:
            pyspice.do_ac_simulation(1e0, 1e5, 101)
            pyspice.plot_bodeplot(block_plots=False)
        elif mode == 3:
            pyspice.do_transient_pulse_simulation(0.0, 1.8, 1e-3, 5e-3, t_sim, fs)
            pyspice.plot_transient(block_plots=False)
        elif mode == 4:
            pyspice.do_transient_sinusoidal_simulation(1.0, 0.25e3, t_sim, fs)
            pyspice.plot_transient(block_plots=False)
        elif mode == 5:
            signal0 = pyspice.create_dummy_signal(t_sim, fs)[1]
            pyspice.do_transient_arbitrary_simulation(signal0, t_sim, fs)
            pyspice.plot_transient(block_plots=False)

    pyspice.print_spice_circuit()
    plt.show(block=True)
