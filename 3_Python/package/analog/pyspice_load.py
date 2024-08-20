import dataclasses
import numpy as np
from PySpice.Spice.Netlist import Circuit

from package.analog.pyspice_handler import PySpice_Handler
from package.analog.dev_load import _generate_signal, _plot_test_results
from package.structure_builder import _create_folder_general_firstrun


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

        _create_folder_general_firstrun()
        self._settings = settings_pyspice

        self._circuit = Circuit("Test")
        self._type_device = self.__init_dev()
        self._type_string = self.__init_dev_string()
        self._type_params = self.__init_params()
        self.__sim_time = 1.0
        self.__prep_model = True

    def __init_dev(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': self._resistor, 'C': self._capacitor, 'L': self._inductor}
        dev_type.update({'Ds': self._diode_single, 'Dd': self._diode_antiparallel})
        dev_type.update({'RDs': self._resistive_schottky_single, 'RDd': self._resistive_schottky_antiparallel})
        dev_type.update({'RaM': self._simple_randles_model})
        return dev_type

    def __init_dev_string(self) -> dict:
        """Initialization of functions to get devices"""
        dev_type = {'R': 'Resistor', 'C': 'Capacitor', 'L': 'Inductor'}
        dev_type.update({'Ds': 'pn-Diode (single)', 'Dd': 'pn-Diode (anti-parallel)'})
        dev_type.update({'RDs': 'Resistive schottky diode (single)', 'RDd': 'Resistive schottky diode (anti-parallel)'})
        dev_type.update({'RaM': 'Simple Randles Model'})
        return dev_type

    def __init_params(self) -> dict:
        """Initialization of Device Parameters"""
        params_dict = {}
        params_dict.update({'R': [self._settings.dev_value], 'C': [self._settings.dev_value], 'L': [self._settings.dev_value]})
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
        if isinstance(du, float) or isinstance(du, int):
            iout = np.zeros((1,), dtype=float)
        else:
            iout = np.zeros(du.shape, dtype=float)

        if self._settings.type in self._type_device.keys():
            self.set_src_mode(True)
            self.load_circuit_model(self._type_device[self._settings.type]())

            if isinstance(du, float):
                results = self.do_dc_simulation(du, self.__sim_time, self._settings.fs_ana)
                iout = results['i_in'][-1]
            else:
                self.set_simulation_duration(du.size / self._settings.fs_ana)
                results = self.do_transient_arbitrary_simulation(du, self.__sim_time, self._settings.fs_ana)
                iout = results['i_in']
                num_dly = iout.size-du.size
                iout = iout[num_dly:]
            self.__prep_model = False
        else:
            print("Error: Model not available - Please check!")
        return np.array(iout)

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
            results = self.get_results(3)
            vout = results['v_in'] + u_inn
            num_dly = vout.size - i_in.size - 1
            vout = vout[num_dly:]
        else:
            print("Error: Model not available - Please check!")
        return vout

    def plot_fit_curve(self, start_value=-5.0, stop_value=+5.0, step_size=0.1, do_ylog=False) -> None:
        """Plotting the output of the polynom fit function
        Args:
            start_value:    Starting point of DC Sweep
            stop_value:     End point of DC Sweep
            step_size:      Step size of DC Sweep
            do_ylog:        Do logarithmic plotting on y-scale
        Returns:
            None
        """
        self.set_src_mode(True)
        self.load_circuit_model(self._type_device[self._settings.type]())

        self.do_dc_sweep_simulation(start_value, stop_value, step_size)
        self.plot_iv_curve(do_log=do_ylog)

    # --------------- OVERVIEW OF MODELS -------------------------
    def _resistor(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Load")
        circuit.R(1, 'input', 'output', params[0])
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _capacitor(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Capacitive Load")
        circuit.C(1, 'input', 'output', params[0])
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _inductor(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Capacitive Load")
        circuit.L(0, 'input', 'output', params[0])
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _diode_single(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Diode")
        circuit.model('myDiode', 'D', IS=params[0], RS=0, N=params[1], VJ=params[2], BV=10, IBV=1e-12)
        circuit.Diode(0, 'input', 'output', model='MyDiode')
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _diode_antiparallel(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Diode")
        circuit.model('myDiode', 'D', IS=params[0], RS=0, N=params[1], VJ=params[2], BV=10, IBV=1e-12)
        circuit.Diode(0, 'input', 'output', model='MyDiode')
        circuit.Diode(1, 'output', 'input', model='MyDiode')
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _resistive_schottky_single(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Diode")
        circuit.model('myDiode', 'D', IS=params[0], RS=0, N=params[1], VJ=params[2], BV=10, IBV=1e-12)
        circuit.R(1, 'input', 'middle', params[3])
        circuit.Diode(0, 'middle', 'output', model='MyDiode')
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _resistive_schottky_antiparallel(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Resistive Diode (Antiparallel)")
        circuit.model('myDiode', 'D', IS=params[0], RS=0, N=params[1], VJ=params[2], BV=10, IBV=1e-12)
        circuit.R(1, 'input', 'middle', params[3])
        circuit.Diode(0, 'middle', 'output', model='MyDiode')
        circuit.Diode(1, 'output', 'middle', model='MyDiode')
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit

    def _simple_randles_model(self) -> Circuit:
        """"""
        params = self._type_params[self._settings.type]
        circuit = Circuit("Simple Randles Model")
        circuit.R(1, 'input', 'middle', params[0])
        circuit.R(2, 'middle', 'output', params[1])
        circuit.C(1, 'middle', 'output', params[2])
        circuit.V('cm', 'output', circuit.gnd, 0.0)
        return circuit


# --------------------- TEST CASE ------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    settings = SettingsPySpice(
        type='R',
        fs_ana=125e3,
        noise_en=False,
        dev_value=20e3,
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
    plt.close('all')
    dev = PySpiceLoad(settings)
    dev.set_simulation_duration(t_end)
    dev.print_types()

    # --- Plotting: I-V curve
    print("\nPlotting I-V curve")
    dev.plot_fit_curve()

    # --- Plotting: Current response
    print("\nPlotting transient current response")
    iout = dev.get_current(uinp, uinn)
    _plot_test_results(t0, uinp - uinn, iout, False, do_ylog)

    # --- Plotting: Voltage response
    print("\nPlotting transient voltage response")
    #uout = dev.get_voltage(iout, uinn)
    #_plot_test_results(t0, uout+uinn, iout, True, do_ylog)
    plt.show(block=True)
