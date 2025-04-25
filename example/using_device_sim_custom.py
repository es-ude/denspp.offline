import matplotlib.pyplot as plt
from denspp.offline.analog.dev_common import generate_signal, plot_test_results
from denspp.offline.analog.dev_load import ElectricalLoad, SettingsDEV


if __name__ == "__main__":
    # --- Declaration of input
    fs_ana = 100e3
    do_ylog = False
    t_end = 0.5e-3
    u_off = 1.35

    t0, uinp = generate_signal(0.5e-3, fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 0.0)
    uinp = 0.125 * uinp + u_off
    uinn = 0.0

    # --------------------- TEST CASE #2: Using Own Class ------------------------------------
    settings = SettingsDEV(
        type='R',
        fs_ana=fs_ana,
        noise_en=False,
        para_en=False,
        dev_value=100e3,
        temp=300
    )

    # --- Model declaration
    plt.close('all')
    dev = ElectricalLoad(settings)
    dev.print_types()

    # --- Plotting: Current response
    print("\nPlotting transient current response")
    iout = dev.get_current(uinp, uinn)
    plot_test_results(t0, uinp - uinn, iout, False, do_ylog)

    # --- Plotting: Voltage response
    print("\nPlotting transient voltage response")
    uout = dev.get_voltage(iout, uinn, u_off, 1e-2)
    plot_test_results(t0, uout + uinn, iout, True, do_ylog)

    # --- Plotting: I-V curve
    print("\nPlotting I-V curve")
    dev.change_boundary_voltage(1.0, 6.0)
    dev.plot_fit_curve()
