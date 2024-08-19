from pyspice_handler import PySpice_Handler, resistor


if __name__ == "__main__":
    # --- Settings
    mode = 5
    do_voltage = False
    fs = 100e3
    t_sim = 20e-3

    circuit = resistor(10e3)

    # --- Definition of Sim mode
    pyspice = PySpice_Handler(input_voltage=do_voltage)
    pyspice.load_circuit_model(circuit)

    if mode == 0:
        results = pyspice.do_dc_simulation(1.0)
    elif mode == 1:
        results = pyspice.do_dc_sweep_simulation(-2.0, 5.0, 1e-3)
        pyspice.plot_iv_curve(do_log=False)
    elif mode == 2:
        results = pyspice.do_ac_simulation(1e0, 1e5, 101)
        pyspice.plot_bodeplot()
    elif mode == 3:
        results = pyspice.do_transient_pulse_simulation(0.0, 1.8, 1e-3, 5e-3, t_sim, fs)
        pyspice.plot_transient()
    elif mode == 4:
        results = pyspice.do_transient_sinusoidal_simulation(1.0, 0.25e3, t_sim, fs)
        pyspice.plot_transient()
    elif mode == 5:
        signal = pyspice.create_dummy_signal(t_sim, fs)[1]
        results = pyspice.do_transient_arbitary_simulation(signal, t_sim, fs)
        pyspice.plot_transient()
    pyspice.print_spice_circuit()
