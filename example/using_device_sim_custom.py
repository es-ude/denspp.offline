import matplotlib.pyplot as plt
from copy import deepcopy
from denspp.offline.analog.dev_handler import generate_test_signal, plot_test_results
from denspp.offline.analog.dev_load import (ElectricalLoad, SettingsDEV, DefaultSettingsDEVResistor,
                                            DefaultSettingsDEVResistiveDiodeSingle, DefaultSettingsDEVResistiveDiodeDouble)


settings = deepcopy(DefaultSettingsDEVResistiveDiodeSingle)


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
    settings.use_poly = True
    dev = ElectricalLoad(settings)
    dev.change_options_fit(
        poly_order=7,
        num_points_fit=101
    )
    dev.print_types()

    # --- Plotting: Current response
    print("\nPlotting transient current response")
    iout = dev.get_current(uinp, uinn)
    plot_test_results(t0, uinp - uinn, iout, False, do_ylog)

    # --- Plotting: Voltage response
    print("\nPlotting transient voltage response")
    uout = dev.get_voltage(iout, uinn)
    plot_test_results(t0, uout + uinn, iout, True, do_ylog)

    # --- Plotting: I-V curve
    print("\nPlotting I-V curve for polynom fitting")
    dev.change_boundary_voltage(0.01, 5.0)
    dev.plot_polyfit_transfer_function()

    print("\nPlotting I-V curve for parameter fitting")
    dev.plot_param_fitting(
        bounds_param={'uth0': [0.05, 0.2], 'i_sat':[0.5e-12, 2e-12], 'n_eff': [1.799, 2.801], 'r_sh': [10e3, 40e3]}
    )
