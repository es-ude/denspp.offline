import numpy as np
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


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
        current[0] = 1.
        return 0


class PySpice_Handler:
    _circuit:            Circuit
    _is_input_voltage:   bool

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

    @property
    def __calc_temp_in_celsius(self) -> float:
        """Translating the temperature value from Kelvin [K] to Grad Celsius [°C]"""
        return self._used_temp - 273.15

    def load_circuit_model(self, circuit: Circuit) -> None:
        """Loading an external circuit SPICE model"""
        self._circuit = circuit

    def print_spice_circuit(self) -> None:
        """Printing the circuit in SPICE format"""
        print(self._circuit)

    def do_dc_simulation(self, value: float, do_print_results=True) -> dict:
        """Performing the DC or Operating Point Simulation
        Args:
            value:              Specified value of the input voltage or current source
            do_print_results:   Printing the node voltages
        Returns:
            Dictionary with node voltages
        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, value)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', 'input', self._circuit.gnd, value)

        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp)
        analysis = simulator.operating_point()

        results = dict()
        for node in analysis.nodes.values():
            results.update({str(node): float(node)})
            if do_print_results:
                unit = 'V'
                print(f'Node {str(node)}: {float(node):4.3f} {unit}')
        return results

    def do_dc_sweep_simulation(self, start_dc: float, stop_dc: float, step_dc: float):
        """Performing the DC or Operating Point Simulation
        Args:
            start_dc:   Starting point of DC Sweep
            stop_dc:    End point of DC Sweep
            step_dc:    Step size of DC Sweep
        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, 0)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', 'input', self._circuit.gnd, 0)

        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp)
        analysis = simulator.dc(Vinput=slice(start_dc, stop_dc, step_dc))
        return analysis

    def do_ac_simulation(self, start_freq: float, stop_freq: float, num_points: int, amplitude=1.0):
        """Performing the DC or Operating Point Simulation
        Args:
            start_freq:     Frequency value for starting point
            stop_freq:      Frequency value for end point
            num_points:     Number of repetions
            amplitude:      Amplitude of input signal [Default: 1.0]
        """
        if self._is_input_voltage:
            self._circuit.SinusoidalVoltageSource('input', 'input', self._circuit.gnd, amplitude)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.SinusoidalCurrentSource('input', 'input', self._circuit.gnd, amplitude)

        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp)
        analysis = simulator.ac(start_frequency=start_freq, stop_frequency=stop_freq,
                                number_of_points=num_points, variation='dec')
        return analysis

    def do_transient_pulse_simulation(self, neg_value: float, pos_value: float,
                                      pulse_width: float, pulse_period: float,
                                      t_sim: float, f_samp: float):
        """Performing the Transient Simulation with Pulse Signal
        Args:
            neg_value:      Pos. peak value of the pulse signal [V or A]
            pos_value:      Neg. peak value of the pulse signal [V or A]
            pulse_width:    Pulse width [s]
            pulse_period:   Period of the pulses [s]
            t_sim:          Total simulation time [s]
            f_samp:         Sampling frequency [Hz]
        Returns:
            None
        """
        if self._is_input_voltage:
            self._circuit.PulseVoltageSource('input', 'input', self._circuit.gnd,
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.PulseCurrentSource('input', 'input', self._circuit.gnd,
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period)

        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp)
        analysis = simulator.transient(step_time=1/f_samp, end_time=t_sim)
        return analysis

    def do_transient_sinusoidal_simulation(self, amp: float, freq: float,
                                           t_sim: float, f_samp: float,
                                           t_dly=0.0, offset=0.0):
        """Performing the Transient Simulation with Sinusoidal Signal
        Args:
            neg_value:      Pos. peak value of the pulse signal [V or A]
            pos_value:      Neg. peak value of the pulse signal [V or A]
            pulse_width:    Pulse width [s]
            pulse_period:   Period of the pulses [s]
            t_sim:          Total simulation time [s]
            f_samp:         Sampling frequency [Hz]
        Returns:
            None
        """
        if self._is_input_voltage:
            self._circuit.SinusoidalVoltageSource(
                'input', 'input', self._circuit.gnd,
                amplitude=amp, delay=t_dly, offset=offset, frequency=freq
            )
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.SinusoidalCurrentSource(
                'input', 'input', self._circuit.gnd,
                amplitude=amp, delay=t_dly, offset=offset, frequency=freq
            )

        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp)
        analysis = simulator.transient(step_time=1/f_samp, end_time=t_sim)
        return analysis

    def do_transient_arbitary_simulation(self, signal: np.ndarray, t_end: float, f_samp: float):
        """Performing the Transient Simulation with Pulse Signal

        """
        if self._is_input_voltage:
            self._circuit.V('input', 'input', self._circuit.gnd, 'dc 0 external')
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', 'input', self._circuit.gnd, 'dc 0 external')

        data = _ArbWFG(signal, f_samp, amplitude=1.0, send_data=False)
        simulator = self._circuit.simulator(temperature=self._used_temp, nominal_temperature=self._used_temp,
                                            simulator='ngspice-shared', ngspice_shared=data)

        simulator.initial_condition(point=0)
        analysis = simulator.transient(step_time=1 / f_samp, end_time=t_end)
        return analysis

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
