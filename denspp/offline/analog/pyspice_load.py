from PySpice.Spice.Netlist import Circuit
from .pyspice_handler import PySpiceHandler, SettingsPySpice


class PySpiceLoad(PySpiceHandler):
    r_shunt: float = 1.

    def __init__(self, settings: SettingsPySpice):
        super().__init__(settings)

        self._register_device(
            short_label='R',
            description='Resistor Circuit',
            func_circ=self._resistor
        )
        self._register_device(
            short_label='L',
            description='Inductor Circuit',
            func_circ=self._inductor
        )
        self._register_device(
            short_label='C',
            description='Capacitive Circuit',
            func_circ=self._capacitor
        )
        self._register_device(
            short_label='Ds',
            description='Diode 1N4148 Circuit',
            func_circ=self._diode_1n4148
        )
        self._register_device(
            short_label='RDs',
            description='Resitive Diode (single) Circuit',
            func_circ=self._resistive_diode
        )
        self._register_device(
            short_label='RDd',
            description='Resitive Diode (anti-parallel) Circuit',
            func_circ=self._resistive_diode_antiparallel
        )
        self._register_device(
            short_label='SR',
            description='Simple Randles Circuit',
            func_circ=self._simple_randles_model
        )
        self._register_device(
            short_label='VD',
            description='Voltage divider Circuit',
            func_circ=self._voltage_divider
        )

    def _resistor(self, r: float=10e3) -> Circuit:
        """PySpice model for handling a resistor in simulation
        :param value: The resistance of the circuit.
        :return: The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Resistive Load")
        circuit0.R(0, 'input', 'output', r)
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _capacitor(self, c: float = 1e-9) -> Circuit:
        """Using capacitor as load element"""
        circuit = Circuit("Capacitive Load")
        circuit.R(0, 'input', 'middle', self.r_shunt)
        circuit.C(0, 'middle', 'output', c)
        circuit.V('cm', 'output', circuit.gnd, self.vcm)
        return circuit

    def _inductor(self, l: float=1e-9) -> Circuit:
        """Using capacitor as load element"""
        circuit = Circuit("Capacitive Load")
        circuit.R(0, 'input', 'middle', self.r_shunt)
        circuit.L(0, 'middle', 'output', l)
        circuit.V('cm', 'output', circuit.gnd, self.vcm)
        return circuit

    def _diode_1n4148(self) -> Circuit:
        """PySpice model for using a 1N4148 diode in simulation
        :return:    The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Diode_1N4148")
        circuit0.model('1N4148', 'D', IS=4.352e-9, N=1.906, BV=110, IBV=1e-5, RS=0.6458,
                       CJO=7.048e-13, V=0.869, M=0.03, FC=0.5, TT=3.48E-9)
        circuit0.Diode(0, 'input', 'middle', model='1N4148')
        circuit0.R(0, 'middle', 'output', self.r_shunt)
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _resistive_diode(self, r0: float=1e3, Uth: float=0.7, IS0: float=4e-12, N: float=2.0) -> Circuit:
        """PySpice model for handling a custom-made diode (series) in simulation
        :param r0:     The resistance of the diode.
        :param Uth:    Threshold voltage of the diode.
        :param IS0:    Saturation current of the diode.
        :param N:      Nonlinear factor of the diode.
        :return:       The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Resistive Diode")
        circuit0.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12, )
        circuit0.R(1, 'input', 'middle', r0 + self.r_shunt)
        circuit0.Diode(0, 'middle', 'output', model='myDiode')
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _resistive_diode_antiparallel(self, r0: float=1e3, Uth: float=0.7, IS0: float=4e-12, N: float=2.0) -> Circuit:
        """PySpice model for handling a custom-made diode (antiparallel) in simulation
        :param r0:     The resistance of the diode.
        :param Uth:    Threshold voltage of the diode.
        :param IS0:    Saturation current of the diode.
        :param N:      Nonlinear factor of the diode.
        :return:       The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Resistive Diode (Antiparallel)")
        circuit0.model('myDiode', 'D', IS=IS0, RS=0, N=N, VJ=Uth, BV=10, IBV=1e-12)
        circuit0.R(1, 'input', 'middle', r0 + self.r_shunt)
        circuit0.Diode(0, 'middle', 'output', model='myDiode')
        circuit0.Diode(1, 'output', 'middle', model='myDiode')
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _simple_randles_model(self, r_tis: float=10e3, r_far: float=100e6, c_dl: float=10e-9) -> Circuit:
        """PySpice model for handling a Randles model in simulation
        :param r_tis:   Tissue resistance.
        :param r_far:   Faraday resistance.
        :param c_dl:    Double layer capacity.
        :return:        The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Simple Randles Model")
        circuit0.R(1, 'input', 'middle', r_tis)
        circuit0.R(2, 'middle', 'output', r_far)
        circuit0.C(1, 'middle', 'output', c_dl)
        circuit0.V('cm', 'output', circuit0.gnd, self.vcm)
        return circuit0

    def _voltage_divider(self, r0: float=10e3, r1: float=10e3, rl: float=10e12, cl: float=0.0) -> Circuit:
        """PySpice model for handling a custom-made diode (series) in simulation
        :param r0:  The resistance of the diode.
        :param r1:  The resistance of the diode.
        :param rl:  The resistance of the diode.
        :param cl:  The resistance of the diode.
        :return:    The circuit model including voltage sources (input, common mode)
        """
        circuit0 = Circuit("Voltage Divider with Load")
        circuit0.R(1, 'input', 'output', r0)
        circuit0.R(2, 'output', 'ref', r1)
        circuit0.R(3, 'output', 'ref', rl)
        if cl:
            circuit0.C(0, 'output', 'ref', cl)
        circuit0.V(0, 'ref', circuit0.gnd, self.vcm)
        return circuit0
