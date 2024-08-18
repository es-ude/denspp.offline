import numpy as np
from pyspice_handler import PySpice_Handler
from pyspice_plot import *
from PySpice.Spice.Netlist import Circuit


if __name__ == "__main__":
    # --- Settings
    mode = 4
    do_voltage = True
    fs = 100e3
    t_sim = 20e-3

    # --- Definition of circuit
    circuit = Circuit('Test')
    circuit.model('myDiode', 'D', IS=4e-12, RS=0, BV=10, IBV=1e-12, N=2, VJ=7, M=0.125)

    # --- Circuit Definition
    circuit.R(1, 'input', 'output', 10e3)
    #circuit.R(2, 'output', 'ref', 100e3)
    #circuit.C(1, 'output', 'ref', 10e-9)
    # circuit.Diode(0, 'output', circuit.gnd, model='MyDiode')
    #circuit.R(2, 'ref', circuit.gnd, 10e3)
    circuit.R(3, 'output', circuit.gnd, 20e3)

    # --- Definition of Sim mode
    pyspice = PySpice_Handler(input_voltage=do_voltage)
    pyspice.load_circuit_model(circuit)
    if mode == 0:
        pyspice.do_dc_simulation(1.0)
    elif mode == 1:
        pyspice.do_dc_sweep_simulation(-2.0, 5.0, 1e-3)
    elif mode == 2:
        results = pyspice.do_ac_simulation(1e0, 1e5, 101)
        freq = np.array(results.frequency)
    elif mode == 3:
        results = pyspice.do_transient_pulse_simulation(0.0, 1.8, 1e-3, 5e-3, t_sim, fs)
        time = np.array(results.time)
    elif mode == 4:
        results = pyspice.do_transient_sinusoidal_simulation(1.0, 0.25e3, t_sim, fs)
        time = np.array(results.time)
    elif mode == 5:
        signal = pyspice.create_dummy_signal(t_sim, fs)[1]
        results = pyspice.do_transient_arbitary_simulation(signal, t_sim, fs)
        time = np.array(results.time)

    # --- Get values
    i_in = np.array(results.branches['vvinput_minus'])
    v_in = np.array(results.input)
    v_out = np.array(results.output)

    # --- Plotten
    if mode == 0:
        pass
    elif mode == 1:
        plot_iv_curve(v_in, i_in, do_log=True)
    elif mode == 2:
        plot_bodeplot(freq, v_in, v_out)
    elif mode == 3:
        plot_transient(time, v_in, v_out, i_in)
    elif mode == 4:
        plot_transient(time, v_in, v_out, i_in)
