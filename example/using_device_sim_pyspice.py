from denspp.offline.analog.pyspice_handler import PySpiceModels, PySpiceHandler


if __name__ == "__main__":
    run_mode = [0, 1, 2, 3, 4, 5]
    do_voltage = True
    fs = 100e3
    t_sim = 20e-3

    models = PySpiceModels()
    circuit = models.voltage_divider(c_load=10e-9)
    # circuit = models.simple_randles_model()
    # circuit = models.diode_1n4148()

    # --- Definition of Sim mode
    pyspice = PySpiceHandler(input_voltage=do_voltage)
    pyspice.get_ngspice_version()
    for mode in run_mode:
        pyspice.load_circuit_model(circuit)
        if mode == 0:
            pyspice.do_dc_simulation(1.0)
        elif mode == 1:
            pyspice.do_dc_sweep_simulation(-12.0, 4.0, 1e-3)
            pyspice.plot_iv_curve(do_log=False)
            pyspice.plot_iv_curve(do_log=True)
        elif mode == 2:
            pyspice.do_ac_simulation(1e0, 1e5, 101)
            pyspice.plot_bodeplot()
        elif mode == 3:
            pyspice.do_transient_pulse_simulation(0.0, 1.8, 1e-3, 5e-3, t_sim, fs)
            pyspice.plot_transient()
        elif mode == 4:
            pyspice.do_transient_sinusoidal_simulation(1.0, 0.25e3, t_sim, fs)
            pyspice.plot_transient()
        elif mode == 5:
            signal0 = pyspice.create_dummy_signal(t_sim, fs)[1]
            pyspice.do_transient_arbitrary_simulation(signal0, t_sim, fs)
            pyspice.plot_transient(show_plot=True)
        pyspice.print_spice_circuit()
