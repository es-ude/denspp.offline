import dataclasses
import numpy as np
from PySpice.Spice.Netlist import Circuit

from package.analog.pyspice_handler import PySpice_Handler
from package.analog.dev_handler import _generate_signal, _plot_test_results
from package.structure_builder import create_folder_general_firstrun


@dataclasses.dataclass
class SettingsPySpice:
    """Individual data class to configure the electrical device
    Inputs:
        type:       Type of electrical device ['R': resistor, 'C': capacitor, 'L': inductor, 'RDs': Resistive diode]
        fs_ana:     Sampling frequency of input [Hz]
        noise_en:   Enable noise on output [True / False]
        dev_value:  Value of the selected electrical device
        temp:       Temperature [K]
    """
    type:       str
    fs_ana:     float
    noise_en:   bool
    dev_value:  float
    temp:       float


RecommendedSettingsDEV = SettingsPySpice(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    dev_value=100e3,
    temp=300
)


class PySpiceLoad(PySpice_Handler):
    _settings: SettingsPySpice
    _type_device: dict
    _type_string: dict
    _type_params: dict
    _params_used: list
    _circuit: Circuit
    __sim_time: float

    def __init__(self, settings_pyspice: SettingsPySpice) -> None:
        super().__init__(settings_pyspice.temp, True)

        create_folder_general_firstrun()
        self._settings = settings_pyspice

        self._circuit = Circuit("Test")
        self._type_device = self.__init_dev()
        self._type_string = self.__init_dev_string()
        self._type_params = self.__init_params()
        self.__sim_time = 1.0
        self.vcm = 0.0

    def __init_dev(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': self._resistor, 'C': self._capacitor}
        dev_type.update({'Ds': self._diode_single, 'Dd': self._diode_antiparallel})
        dev_type.update({'RDs': self._resistive_diode_single, 'RDd': self._resistive_diode_antiparallel})
        dev_type.update({'RaM': self._simple_randles_model})
        return dev_type

    def __init_dev_string(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': 'Resistor', 'C': 'Capacitor'}
        dev_type.update({'Ds': 'pn-Diode (single)', 'Dd': 'pn-Diode (anti-parallel)'})
        dev_type.update({'RDs': 'Resistive diode (single)', 'RDd': 'Resistive diode (anti-parallel)'})
        dev_type.update({'RaM': 'Simple Randles Model'})
        return dev_type

    def __init_params(self) -> dict:
        """Initialization of Device Parameters"""
        params_dict = {}
        params_dict.update({'R': [self._settings.dev_value], 'C': [self._settings.dev_value]})
        params_dict.update({'Ds': [1e-12, 1.4, 0.7], 'Dd': [1e-12, 1.4, 0.7]})
        params_dict.update({'RDs': [1e-12, 2.8, 0.1, self._settings.dev_value]})
        params_dict.update({'RDd': [1e-12, 2.8, 0.1, self._settings.dev_value]})
        params_dict.update({'RaM': [10e3, 100e6, 10e-9]})
        return params_dict

    def set_simulation_duration(self, sim_time: float) -> None:
        """Defining the simulation duration for SPICE simulation"""
        self.__sim_time = sim_time

    def print_types(self) -> None:
        """Print electrical types in terminal"""
        print("\n==========================================="
              "\nAvailable types of electrical devices")
        for idx, type in enumerate(self._type_device.keys()):
            print(f"\t#{idx:03d}: {type} = {self._type_string[type]}")

    def get_current(self, u_top: np.ndarray | float, u_bot: np.ndarray | float) -> np.ndarray:
        """Getting the current response from electrical device
        Args:
            u_top:      Applied voltage on top electrode [V]
            u_bot:      Applied voltage on bottom electrode  [V]
        Returns:
            Corresponding current response
        """
        du = u_top - u_bot
        if self._settings.type in self._type_device.keys():
            self.set_src_mode(True)
            self.load_circuit_model(self._type_device[self._settings.type]())

            if isinstance(du, float) or isinstance(du, int):
                results = self.do_dc_simulation(du, do_print_results=False)
                i_out0 = results['i_in']
            else:
                self.set_simulation_duration(du.size / self._settings.fs_ana)
                results = self.do_transient_arbitrary_simulation(du, self.__sim_time, self._settings.fs_ana)
                i_out0 = results['i_in']
                num_dly = i_out0.size-du.size-1
                i_out0 = i_out0[num_dly:-1]
        else:
            raise "Error: Model not available - Please check!"
        return np.array(i_out0)

    def get_voltage(self, i_in: np.ndarray, u_inn: np.ndarray | float) -> np.ndarray:
        """Getting the voltage response from electrical device
        Args:
            i_in:               Applied current input [A]
            u_inn:              Negative input | bottom electrode | reference voltage [V]
        Returns:
            Corresponding voltage response
        """
        if isinstance(i_in, float) or isinstance(u_inn, int):
            vout = np.zeros((1,), dtype=float)
        else:
            vout = np.zeros(i_in.shape, dtype=float)

        if self._settings.type in self._type_device.keys():
            self.set_src_mode(False)
            self.set_simulation_duration(i_in.size / self._settings.fs_ana)
            self.load_circuit_model(self._type_device[self._settings.type]())

            self.do_transient_arbitrary_simulation(i_in, self.__sim_time, self._settings.fs_ana)
            results = self.__get_results(3)
            vout = results['v_in'] + u_inn
            num_dly = vout.size - i_in.size - 1
            vout = vout[num_dly:]
        else:
            print("Error: Model not available - Please check!")
        return vout

    def plot_fit_curve(self, start_value=-5.0, stop_value=+5.0, step_size=0.1, do_logy=False,
                       path2save='', show_plot=False) -> None:
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
        self.load_circuit_model(self._type_device[self._settings.type]())

        self.do_dc_sweep_simulation(start_value, stop_value, step_size)
        self.plot_iv_curve(do_logy, path2save, show_plot)

    # --------------- OVERVIEW OF MODELS -------------------------
    def _resistor(self) -> Circuit:
        """Using resistor as load element"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Load")
        circuit.R(1, 'input', 'output', params[0])
        circuit.V('cm', 'output', circuit.gnd, self.vcm)
        return circuit

    def _capacitor(self) -> Circuit:
        """Using capacitor as load element"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Capacitive Load")
        circuit.R(0, 'input', 'middle', 100)
        circuit.C(0, 'middle', 'output', params[0])
        circuit.V('cm', 'output', circuit.gnd, self.vcm)
        return circuit

    def _diode_single(self) -> Circuit:
        """Using 1N4148 diode as load element"""
        params = self._type_params[self._settings.type]
        circuit0 = Circuit("Diode_1N4148")
        circuit0.model('1N4148', 'D', IS=4.352e-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458,
                       CJO=7.048e-13, V=0.869, M=0.03, FC=0.5, TT=3.48E-9)
        circuit0.R(0, 'input', 'middle', 10)
        circuit0.Diode(0, 'middle', 'output', model='1N4148')
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _diode_antiparallel(self) -> Circuit:
        """Using 1N4148 in anti-parallel configuration as load element"""
        params = self._type_params[self._settings.type]
        circuit0 = Circuit("Diode_1N4148 (Antiparallel)")
        circuit0.model('1N4148', 'D', IS=4.352e-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458,
                       CJO=7.048e-13, V=0.869, M=0.03, FC=0.5, TT=3.48E-9)
        circuit0.Diode(0, 'input', 'middle', model='1N4148')
        circuit0.Diode(1, 'middle', 'input', model='1N4148')
        circuit0.R(0, 'middle', 'output', 10)
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _resistive_diode_single(self) -> Circuit:
        """Using resistive diode as load element"""
        params = self._type_params[self._settings.type]
        circuit0 = Circuit("Resistive Diode")
        circuit0.model('1N4148', 'D', IS=4.352e-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458,
                       CJO=7.048e-13, V=0.869, M=0.03, FC=0.5, TT=3.48E-9)
        circuit0.R(0, 'input', 'middle', params[3])
        circuit0.Diode(0, 'middle', 'output', model='1N4148')
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _resistive_diode_antiparallel(self) -> Circuit:
        """Using resistive diode in anti-parallel configuration as load element"""
        params = self._type_params[self._settings.type]
        circuit0 = Circuit("Resistive Diode (Antiparallel)")
        circuit0.model('1N4148', 'D', IS=4.352e-9, N=1.906, BV=110, IBV=0.0001, RS=0.6458,
                       CJO=7.048e-13, V=0.869, M=0.03, FC=0.5, TT=3.48E-9)
        circuit0.R(0, 'input', 'middle', params[3])
        circuit0.Diode(0, 'middle', 'output', model='1N4148')
        circuit0.Diode(1, 'output', 'middle', model='1N4148')
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _simple_randles_model(self) -> Circuit:
        """Using simple Randles model as load element"""
        params = self._type_params[self._settings.type]
        circuit0 = Circuit("Simple Randles Model")
        circuit0.R('tis', 'input', 'middle', params[0])
        circuit0.R('far', 'middle', 'output', params[1])
        circuit0.C('dl', 'middle', 'output', params[2])
        circuit0.V('cm', 'output', circuit0.gnd, 0.0)
        return circuit0


# --------------------- TEST CASE ------------------------------------
if __name__ == "__main__":
    settings = SettingsPySpice(
        type='R',
        fs_ana=500e3,
        noise_en=False,
        dev_value=100e3,
        temp=300
    )

    # --- Declaration of input
    do_ylog = False
    t_end = 0.5e-3
    u_off = 1.35

    t0, uinp = _generate_signal(0.5e-3, settings.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 0.0)
    uinp = 0.125 * uinp + u_off
    uinn = 0.0

    # --- Model declaration
    dev = PySpiceLoad(settings)
    dev.set_simulation_duration(t_end)
    dev.print_types()

    # --- Plotting: Current response
    print("\nPlotting transient current response")
    iout = dev.get_current(uinp, uinn)
    _plot_test_results(t0, uinp - uinn, iout, False, do_ylog)

    # --- Plotting: Voltage response
    print("\nPlotting transient voltage response")
    uout = dev.get_voltage(iout, uinn)
    _plot_test_results(t0, uout+uinn, iout, True, do_ylog)

    # --- Plotting: I-V curve
    print("\nPlotting I-V curve")
    dev.plot_fit_curve(do_logy=do_ylog, show_plot=True)
