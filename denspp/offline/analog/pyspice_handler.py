import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from inspect import getfullargspec
from logging import getLogger, Logger
from PySpice.Spice.Netlist import Circuit, Netlist
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from denspp.offline.plot_helper import scale_auto_value, save_figure


def _add_method(o: object, method, name) -> None:
    """Changing the functionality of an attribute"""
    setattr(o, name, method.__get__(o, type(o)))


def _create_arbfwg(spice_instance: NgSpiceShared, waveform: np.ndarray, f_samp: float) -> None:
    """Bugfixing for replacement a PySPICE function"""
    def get_vsrc_data(self, voltage, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        index = int(time * f_samp) % len(waveform)
        voltage[0] = waveform[index]
        return 0

    _add_method(spice_instance, get_vsrc_data, "get_vsrc_data")


def _clear_arbfwg(spice_instance: NgSpiceShared) -> None:
    """Function for getting the voltage source value for making transient analysis"""
    def get_vsrc_data(self, voltage, time, node, ngspice_id) -> int:
        """Internal NgSpice function for simulation"""
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        return 0

    _add_method(spice_instance, get_vsrc_data, "get_vsrc_data")


def do_bugfix_clone_circuit(circuits_netlist: Netlist) -> None:
    """Bugfix function for cloning circuits successful"""
    def copy_to(self, netlist: Netlist) -> Netlist:
        for subcircuit in self.subcircuits:
            netlist.subcircuit(subcircuit)

        for element in self.elements:
            element.copy_to(netlist)

        for name, model in self._models.items():
            netlist._models[name] = model

        netlist.raw_spice = str(self.raw_spice)
        return netlist

    _add_method(circuits_netlist, copy_to, "copy_to")


@dataclass
class SettingsPySpice:
    """Individual data class to configure the electrical device
    Attributes:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor, 'RDs': Resistive diode]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        params_use: Dictionary with used parameters of the circuit/model
        temp_kelvin:Temperature [K]
        input_volt: Boolean if input is voltage (True) or current (False)
    """
    type:           str
    fs_ana:         float
    noise_en:       bool
    params_use:     dict
    temp_kelvin:    float
    input_volt:     bool

    @property
    def temp_celsius(self) -> float:
        """Translating the temperature value from Kelvin [K] to Grad Celsius [°C]"""
        return self.temp_kelvin - 273.15


RecommendedSettingsDEV = SettingsPySpice(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    params_use={'r': 100e3},
    temp_kelvin=300,
    input_volt=True
)


############################################################################
class PySpiceHandler:
    _logger: Logger
    _type_device: dict = dict()
    _circuit: Circuit = Circuit("Test")
    _results: dict = {}
    _run_ite: int = 0
    _sim_time: float = 1.0
    vcm: float = 0.0
    _settings: SettingsPySpice
    _results: dict = dict()

    def __init__(self, settings: SettingsPySpice) -> None:
        """Rewritten API for using PySPICE in simulation
        (Git Tutorial: https://github.com/benedictjones/engineeringthings-pyspice,
        YouTube: https://www.youtube.com/watch?v=62BOYx1UCfs&list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh)
        Args:
            settings:    Dataclass SettingsPySpice with settings for simulation
        Returns:
            None
        """
        self._logger = getLogger(__name__)
        self._settings = settings

    def _register_device(self, short_label: str, description: str, func_circ) -> None:
        """Function for registering an electrical device to library
        :param short_label:     String with short label of the device (e.g. 'R')
        :param description:     Short description of device type (e.g. 'Resistor')
        :param func_circ:       Function to the implemented SPICE circuit for operation
        """
        param = getfullargspec(func_circ)[0][1:]
        self._logger.debug(f"Registering device: {short_label} ({description}) with params: {param}")
        self._type_device.update({short_label: {'desp': description, 'param': param, 'circ': func_circ}})

    def _load_circuit_model(self) -> None:
        """Loading an external circuit SPICE model"""
        if self._settings.type in self._type_device.keys():
            if [key for key in self._settings.params_use.keys()] == self._type_device[self._settings.type]['param']:
                circuit_loaded = self._type_device[self._settings.type]['circ'](**self._settings.params_use)
            else:
                raise NotImplementedError(f"Model Params are unequal: Need {self._type_device[self._settings.type]['param']} is not supported")
        else:
            raise NotImplementedError(f"Type {self._settings.type} is not supported (only {self._type_device.keys()})")

        self._circuit = Circuit(circuit_loaded.title)
        do_bugfix_clone_circuit(circuit_loaded)
        circuit_loaded.copy_to(self._circuit)

    @staticmethod
    def get_ngspice_version() -> str:
        """Getting the version of used NGspice in PySPICE"""
        from PySpice import __version__
        from PySpice.Spice.NgSpice import NGSPICE_SUPPORTED_VERSION
        return f"PySpice v{__version__} with NGSpice v{NGSPICE_SUPPORTED_VERSION}"

    def set_simulation_duration(self, sim_time: float) -> None:
        """Defining the simulation duration for SPICE simulation"""
        self._sim_time = sim_time

    def set_src_mode(self, do_voltage: bool) -> None:
        """Setting the Source Mode of Input Source [0: Current, 1: Voltage]"""
        self._settings.input_volt = do_voltage

    def print_spice_circuit(self) -> str:
        """Printing the circuit in SPICE format"""
        self._logger.info("======================================================")
        self._logger.info("\tCIRCUIT SPICE IMPLEMENTATION")
        self._logger.info("======================================================")
        self._logger.info(self._circuit)
        return str(self._circuit)

    def print_types(self) -> list:
        """Print electrical types in terminal"""
        self._logger.info("===========================================")
        self._logger.info("\tAvailable types of electrical devices")
        self._logger.info("===========================================")
        methods = list()
        for idx, type in enumerate(self._type_device.keys()):
            self._logger.info(f"\t#{idx:03d}: {type} = {self._type_device[type]['desp']}")
            methods.append(f"{type} = {self._type_device[type]['desp']} (params = {self._type_device[type]['param']})")
        return methods

    ############################################################################

    def do_dc_simulation(self, value: float, initial_value: float=0.0) -> dict:
        """Performing the DC or Operating Point Simulation
        Args:
            value:              Specified value of the input voltage or current source
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            Dictionary with node voltages
        """
        self._load_circuit_model()
        if self._settings.input_volt:
            self._circuit.V('input', 'input', self._circuit.gnd, value)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', value)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius, nominal_temperature=self._settings.temp_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        analysis = simulator.operating_point()

        results = dict()
        for node in analysis.nodes.values():
            results.update({str(node): float(node)})

        return self.__get_results(0, analysis)

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
        Returns:
            Corresponding current response
        """
        self._load_circuit_model()
        du = u_top - u_bot
        self.set_src_mode(True)

        if isinstance(du, float) or isinstance(du, int):
            results = self.do_dc_simulation(du, initial_value=0.0)
            i_out0 = results['i_in']
        else:
            self.set_simulation_duration(du.size / self._settings.fs_ana)
            results = self.do_transient_arbitrary_simulation(du, self._sim_time, self._settings.fs_ana)
            i_out0 = results['i_in']
            num_dly = i_out0.size-du.size-1
            i_out0 = i_out0[num_dly:-1]
        return np.array(i_out0)

    def get_voltage(self, i_in: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Getting the voltage response from electrical device
        Args:
            i_in:               Applied current input [A]
            u_inn:              Negative input | bottom electrode | reference voltage [V]
        Returns:
            Corresponding voltage response
        """
        self._load_circuit_model()
        if isinstance(i_in, float) or isinstance(u_inn, int):
            vout = np.zeros((1,), dtype=float)
        else:
            vout = np.zeros(i_in.shape, dtype=float)

        self.set_src_mode(False)
        self.set_simulation_duration(i_in.size / self._settings.fs_ana)
        results = self.do_transient_arbitrary_simulation(i_in, self._sim_time, self._settings.fs_ana)
        vout = results['v_in'] + u_inn
        num_dly = vout.size - i_in.size - 1
        vout = vout[num_dly:]
        return vout

    ############################################################################

    def do_dc_sweep_simulation(self, start_dc: float, stop_dc: float, step_dc: float, initial_value: float=0.0) -> dict:
        """Performing the DC or Operating Point Simulation
        Args:
            start_dc:   Starting point of DC Sweep
            stop_dc:    End point of DC Sweep
            step_dc:    Step size of DC Sweep
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        self._load_circuit_model()
        if self._settings.input_volt:
            self._circuit.V('input', 'input', self._circuit.gnd, 0)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.I('input', self._circuit.gnd, 'input', 0)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius, nominal_temperature=self._settings.temp_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)

        if self._settings.input_volt:
            results = simulator.dc(Vinput=slice(start_dc, stop_dc, step_dc))
        else:
            results = simulator.dc(Iinput=slice(start_dc, stop_dc, step_dc))

        self._results = self.__get_results(1, results)
        return self._results

    ############################################################################

    def do_ac_simulation(self, start_freq: float, stop_freq: float, num_points: int,
                         amplitude: float=1.0, initial_value: float=0.0) -> dict:
        """Performing the DC or Operating Point Simulation
        Args:
            start_freq:     Frequency value for starting point
            stop_freq:      Frequency value for end point
            num_points:     Number of repetitions
            amplitude:      Amplitude of input signal [Default: 1.0]
            initial_value: Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        self._load_circuit_model()
        if self._settings.input_volt:
            self._circuit.SinusoidalVoltageSource('input', 'input', self._circuit.gnd, amplitude)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.SinusoidalCurrentSource('input', self._circuit.gnd, 'input', amplitude)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius, nominal_temperature=self._settings.temp_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.ac(start_frequency=start_freq, stop_frequency=stop_freq,
                               number_of_points=num_points, variation='dec')
        self._results = self.__get_results(2, results)
        return self._results

    ############################################################################

    def do_transient_pulse_simulation(self, neg_value: float, pos_value: float,
                                      pulse_width: float, pulse_period: float,
                                      t_sim: float, initial_value: float=0.0) -> dict:
        """Performing the Transient Simulation with Pulse Signal
        Args:
            neg_value:      Pos. peak value of the pulse signal [V or A]
            pos_value:      Neg. peak value of the pulse signal [V or A]
            pulse_width:    Pulse width [s]
            pulse_period:   Period of the pulses [s]
            t_sim:          Total simulation time [s]
            initial_value:  Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        self._load_circuit_model()
        if self._settings.input_volt:
            self._circuit.PulseVoltageSource('input', 'input', self._circuit.gnd,
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period,
                                             rise_time=1 / self._settings.fs_ana, fall_time=1 / self._settings.fs_ana)
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.PulseCurrentSource('input', self._circuit.gnd, 'input',
                                             initial_value=neg_value, pulsed_value=pos_value,
                                             pulse_width=pulse_width, period=pulse_period,
                                             rise_time=1 / self._settings.fs_ana, fall_time=1 / self._settings.fs_ana)
            self._circuit.Iinput.plus.add_current_probe(self._circuit)

        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius, nominal_temperature=self._settings.temp_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.transient(step_time=1 / self._settings.fs_ana, end_time=t_sim)

        self._results = self.__get_results(3, results)
        return self._results

    ############################################################################

    def do_transient_sinusoidal_simulation(self, amp: float, freq: float, t_sim: float,
                                           t_dly: float=0.0, offset: float=0.0, initial_value: float=0.0) -> dict:
        """Performing the Transient Simulation with Sinusoidal Signal
        Args:
            amp:            Amplitude of sinusoidal waveform [V or A]
            freq:           Frequency of sinusoidal waveform [Hz]
            t_sim:          Total simulation time [s]
            t_dly:          Applied time delay to signal [s] [Default: 0.0]
            offset:         Applied offset on signal [V or A] [Default: 0.0]
            initial_value:  Applied initial value [Default: 0.0]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        self._load_circuit_model()
        if self._settings.input_volt:
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

        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius, nominal_temperature=self._settings.temp_celsius)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.transient(step_time=1 / self._settings.fs_ana, end_time=t_sim)

        self._results = self.__get_results(3, results)
        return self._results

    ############################################################################

    def do_transient_arbitrary_simulation(self, signal: np.ndarray, t_end: float,
                                          initial_value: float=0.0, trans_value: float=1.0) -> dict:
        """Performing the Transient Simulation with Arbitrary Signal Waveform
        Args:
            signal:         Numpy array with transient custom-made signal
            t_end:          Total simulation time [s]
            initial_value:  Applied initial value [Default: 0.0]
            trans_value:    Transcondunctance value [Default: 1 A/V]
        Returns:
            NGSpice dictionary with simulation results [optional]
        """
        arbitrary_signal = NgSpiceShared.new_instance()

        self._load_circuit_model()
        # --- Definition of energy source
        if self._settings.input_volt:
            self._circuit.V('input', 'input', self._circuit.gnd, 'dc 0 external')
            self._circuit.Vinput.minus.add_current_probe(self._circuit)
        else:
            self._circuit.V('data', 'src', self._circuit.gnd, 'dc 0 external')
            self._circuit.VCCS('input', self._circuit.gnd, 'input0', 'src', self._circuit.gnd, trans_value)
            self._circuit.R('sens', 'input0', 'input', 0.0)
            self._circuit.Rsens.plus.add_current_probe(self._circuit)

        # --- Generating instance for using arbitrary waveforms in SPICE transient simulation
        _create_arbfwg(arbitrary_signal, signal, self._settings.fs_ana)
        data = arbitrary_signal

        # --- Prepare and Run simulation
        simulator = self._circuit.simulator(temperature=self._settings.temp_celsius,
                                            nominal_temperature=self._settings.temp_celsius,
                                            simulator='ngspice-shared', ngspice_shared=data)
        if initial_value:
            simulator.initial_condition(point=initial_value)
        results = simulator.transient(step_time=1 / self._settings.fs_ana, end_time=t_end)

        # --- Process results
        _clear_arbfwg(arbitrary_signal)
        self._results = self.__get_results(4, results)
        return self._results

    ############################################################################

    def __get_results(self, mode: int, data) -> dict:
        """Getting the results from already runned SPICE analysis
        Args:
            mode:   Selection mode for getting results (0 = Operating Point, 1 = DC Sweep, 2 = AC Sweep, 3 = Transient)
        Returns:
              Dictionary with entries "v_in", "v_out", "i_in", "time", "freq"
        """
        output_dict = dict()
        cur_in_key0 = 'vvinput_minus' if self._settings.input_volt else 'viinput_plus'
        cur_in_key1 = 'vvinput_minus' if self._settings.input_volt else 'vrsens_plus'
        match mode:
            case 0:
                # DC Operating Point Analysis
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key0])})
            case 1:
                # DC Sweep Analysis
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key0])})
            case 2:
                # AC Sweep Analysis
                output_dict.update({"freq": np.array(data.frequency)})
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key0])})
                output_dict.update({"v_out": np.array(data.output)})
            case 3:
                # Transient Simulation (Pulse, Sinusoidal)
                output_dict.update({"time": np.array(data.time)})
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key0])})
                output_dict.update({"v_out": np.array(data.output)})
            case 4:
                # Transient Simulation (Pulse, Sinusoidal)
                output_dict.update({"time": np.array(data.time)})
                output_dict.update({"v_in": np.array(data.input)})
                output_dict.update({"i_in": np.array(data.branches[cur_in_key1])})
                output_dict.update({"v_out": np.array(data.output)})
        return output_dict

    ############################################################################

    def plot_iv_curve(self, do_log: bool=False, path2save: str='', show_plot: bool=False) -> None:
        """Plotting the I-V relationship/curve of investigated circuit (taking v_in and i_in)
        Args:
            do_log:         Do a logarithmic plotting on y-axis
            path2save:      Optional string for plotting [Default: '' for non-plotting]
            show_plot:      Blocking plots for showing [Default: False]
        Returns:
          None
        """
        # --- Getting data
        results = self._results
        u_in = results["v_in"]
        scale_u, units_u = scale_auto_value(u_in)
        i_out = results["i_in"]
        scale_i, units_i = scale_auto_value(i_out)

        # --- Plotting
        plt.figure()
        if not do_log:
            plt.plot(scale_u * u_in, scale_i * i_out, 'k', linewidth=1, marker='.')
        else:
            plt.semilogy(scale_u * u_in, scale_i * np.abs(i_out), 'k', linewidth=1, marker='.')
        plt.xlabel(fr'Voltage $U$ / {units_u}V')
        plt.ylabel(fr'Current $I$ / {units_i}A')
        plt.xlim([scale_u * u_in[0], scale_u * u_in[-1]])

        plt.tight_layout()
        plt.grid()
        if path2save:
            save_figure(plt, path2save, 'pyspice_dc_result', formats=['svg'])
        if show_plot:
            plt.show(block=True)

    def plot_bodeplot(self, mode: int=0, path2save: str='', show_plot: str=False) -> None:
        """Plotting the Bode Diagram (mode == 0) or Impedance Plot (mode == 1) of investigated circuit
        Args:
            mode:           Mode selection (0 = Bode diagram, 1 = Impedance plot)
            path2save:      Optional string for plotting [Default: '' for non-plotting]
            show_plot:      Blocking plots for showing [Default: False]
        Returns:
          None
        """
        # --- Getting data
        results = self._results
        freq = results["freq"]
        transfer_function = results["v_out"] / results["v_in"] if mode == 0 else results["v_in"] / results["i_in"]

        # --- Plotting
        fig, axs = plt.subplots(2, 1, sharex="all")
        axs[0].semilogx(freq, 20 * np.log10(np.abs(transfer_function)), 'k', linewidth=1, marker='.', label="Gain")
        axs[0].set_ylabel(r"Gain $v_U$ / dB")
        axs[0].set_xlim([freq[0], freq[-1]])

        axs[1].semilogx(freq, np.angle(transfer_function, deg=True), 'r', linewidth=1, marker='.', label="Phase")
        axs[1].set_ylabel(r"Phase $\alpha$ / °")
        axs[1].set_xlabel(r'Frequency $f$ / Hz')
        axs[1].set_xlim([freq[0], freq[-1]])

        for ax in axs:
            ax.set_xlim([freq[0], freq[-1]])
            ax.grid(which='both', linestyle='--')

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.05)
        if path2save:
            save_figure(plt, path2save, 'pyspice_ac_result', ['svg'])
        if show_plot:
            plt.show(block=True)

    def plot_transient(self, path2save: str='', show_plot: bool=False) -> None:
        """Plotting the results of Transient Simulation of investigated circuit
        Args:
            path2save:  Optional string for plotting [Default: '' for non-plotting]
            show_plot:  Blocking plots for showing [Default: False]
        Returns:
          None
        """
        # --- Getting data
        results = self._results
        time = results["time"]
        scale_t, units_t = scale_auto_value(time)
        v_sig = [results["v_in"], results["v_out"]]
        scale_u, units_u = scale_auto_value(np.concatenate((v_sig[0], v_sig[1]), axis=0))
        v_label = ["Input", "Output"]
        i_in = results["i_in"]
        scale_i, units_i = scale_auto_value(i_in)

        # --- Plotting
        fig, ax1 = plt.subplots()
        idx = 0
        for sig, label in zip(v_sig, v_label):
            if idx == 0:
                lns = ax1.plot(scale_t * time, scale_u * sig, linewidth=1, label=label)
            else:
                lns += ax1.plot(scale_t * time, scale_u * sig, linewidth=1, label=label)
            idx += 1

        ax1.set_ylabel(fr"Voltage $U_x$ / {units_u}V")
        ax1.set_xlabel(fr'Time $t$ / {units_t}s')

        ax2 = ax1.twinx()
        lns += ax2.plot(scale_t * time, scale_i * i_in, 'r--', linewidth=1, label='Current')
        ax2.set_ylabel(fr"Current $I$ / {units_i}A")
        ax2.set_xlim([scale_t * time[0], scale_t * time[-1]])

        # added these three lines
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0)

        plt.tight_layout()
        plt.grid()
        if path2save:
            save_figure(plt, path2save, 'pyspice_transient_result', ['svg'])
        if show_plot:
            plt.show(block=True)

    def plot_fit_curve(self, start_value: float=-5.0, stop_value: float=+5.0, step_size: float=0.1,
                       do_logy: bool=False, path2save: str='', show_plot: bool=False) -> None:
        """Plotting the output of the polynom fit function
        Args:
            start_value:    Starting point of DC Sweep
            stop_value:     End point of DC Sweep
            step_size:      Step size of DC Sweep
            do_logy:        Do logarithmic plotting on y-scale
            path2save:      Path for saving the plot
            show_plot:      Showing and blocking the plot
        Returns:
            None
        """
        self.set_src_mode(True)
        self._load_circuit_model(self._type_device[self._settings.type]())

        self.do_dc_sweep_simulation(start_value, stop_value, step_size)
        self.plot_iv_curve(do_logy, path2save, show_plot)


def create_dummy_signal(t_sim: float, f_samp: float, offset: float=0.0, freq_used: list=[100, 300, 500], freq_amp: list=[1.0, 0.25, 0.66]) -> [np.ndarray, np.ndarray]:
    """Creating a dummy function for transient simulation
    Args:
        t_sim:  Simulation time [s]
        f_samp: Sampling frequency [Hz]
        offset: Offset on signal [Default: 0.0]
    Returns:
        Two numpy arrays with time vector and signal vector
    """
    time0 = np.linspace(0, t_sim, int(t_sim * f_samp), endpoint=True)
    sig_out = np.zeros(time0.shape) + offset
    for freq, amp in zip(freq_used, freq_amp):
        sig_out += amp * np.sin(2 * np.pi * freq * time0)
    return time0, sig_out