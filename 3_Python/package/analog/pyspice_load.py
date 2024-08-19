from PySpice.Spice.Netlist import Circuit


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

