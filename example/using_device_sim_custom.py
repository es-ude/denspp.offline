import matplotlib.pyplot as plt
from denspp.offline.analog.dev_handler import generate_test_signal, plot_test_results
from denspp.offline.analog.dev_load import ElectricalLoad, SettingsDEV, DefaultSettingsDEVResistiveDiodeSingle, DefaultSettingsDEVResistiveDiodeDouble


settings = SettingsDEV(
    type='R',
    fs_ana=100e3,
    noise_en=False,
    para_en=False,
    dev_value={'r': 100e3},
    temp=300,
    use_mode=0
)


if __name__ == "__main__":
    # --- Declaration of input
    do_ylog = False
    t_end = 0.5e-3
    u_off = 1.35

    t0, uinp = generate_test_signal(0.5e-3, settings.fs_ana, [2.5, 0.3, 0.1], [10e3, 18e3, 28e3], 0.0)
    uinp = 0.125 * uinp + u_off
    uinn = 0.0


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
    dev.plot_polyfit_tranfer_function()
