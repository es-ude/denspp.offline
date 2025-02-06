import numpy as np
import matplotlib.pyplot as plt

from denspp.offline.plot_helper import scale_auto_value, save_figure
from denspp.offline.analog.dev_load import ElectricalLoad, SettingsDEV
from denspp.offline.analog.pyspice_load import PySpiceLoad, SettingsPySpice


# ================================ FUNCTIONS FOR TESTING ===================================
def generate_signal(t_end: float, fs: float, upp: list, fsig: list, uoff: float=0.0) -> [np.ndarray, np.ndarray]:
    """Generating a signal for testing
    Args:
        t_end:      End of simulation
        fs:         Sampling rate
        upp:        List with amplitude values
        fsig:       List with corresponding frequency
        uoff:       Offset voltage
    Returns:
        List with two numpy arrays (time, voltage signal)
    """
    t0 = np.arange(0, t_end, 1/fs)
    uinp = np.zeros(t0.shape) + uoff
    for idx, peak_val in enumerate(upp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * fsig[idx])
    return t0, uinp


def plot_test_results(time: np.ndarray, u_in: np.ndarray, i_in: np.ndarray,
                      mode_current_input: bool, do_ylog: bool=False, plot_gray: bool=False,
                      path2save: str='', show_plot: bool=False) -> None:
    """Function for plotting transient signal and I-V curve of the used electrical device
    Args:
        time:       Numpy array with time information
        u_in:       Numpy array with input voltage (mode_current_input = False) or output voltage (True)
        i_in:       Numpy array with output current (mode_current_input = False) or input current (True)
        mode_current_input: Bool decision for selecting right source and sink value
        do_ylog:    Plotting the current in the I-V-curve normal (False) or logarithmic (True)
        plot_gray:  Plotting the response of device in red dashed (False) or gray dashed (True)
        path2save:  Path for saving the plot
        show_plot:  Showing and blocking the plot
    Returns:
        None
    """
    scale_i, units_i = scale_auto_value(i_in)
    scale_u, units_u = scale_auto_value(u_in)
    scale_t, units_t = scale_auto_value(time)

    signalx = scale_i * i_in if mode_current_input else scale_u * u_in
    signaly = scale_u * u_in if mode_current_input else scale_i * i_in
    label_axisx = f'Voltage U_x [{units_u}V]' if mode_current_input else f'Current I_x [{units_i}A]'
    label_axisy = f'Current I_x [{units_i}A]' if mode_current_input else f'Voltage U_x [{units_u}V]'
    label_legx = 'i_in' if mode_current_input else 'u_in'
    label_legy = 'u_out' if mode_current_input else 'i_out'

    # --- Plotting: Transient signals
    plt.figure()
    num_rows = 2
    axs = [plt.subplot(num_rows, 1, idx + 1) for idx in range(num_rows)]

    axs[0].set_xlim(scale_t * time[0], scale_t * time[-1])
    twin1 = axs[0].twinx()
    a = axs[0].plot(scale_t * time, signalx, 'k', label=label_legx)
    axs[0].set_ylabel(label_axisy)
    axs[0].set_xlabel(f'Time t [{units_t}s]')
    if plot_gray:
        b = twin1.plot(scale_t * time, signaly, linestyle='dashed', color=[0.5, 0.5, 0.5], label=label_legy)
    else:
        b = twin1.plot(scale_t * time, signaly, 'r--', label=label_legy)
    twin1.set_ylabel(label_axisx)
    axs[0].grid()

    # Generate common legend
    lns = a + b
    labs = [l.get_label() for l in lns]
    axs[0].legend(lns, labs, loc=0)

    # --- Plotting: I-U curve
    if mode_current_input:
        if do_ylog:
            axs[1].semilogy(signaly, signalx, 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signaly, signalx, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisx)
        axs[1].set_ylabel(label_axisy)
    else:
        if do_ylog:
            axs[1].semilogy(signalx, abs(signaly), 'k', marker='.', linestyle='None')
        else:
            axs[1].plot(signalx, signaly, 'k', marker='.', linestyle='None')
        axs[1].set_xlabel(label_axisy)
        axs[1].set_ylabel(label_axisx)
    axs[1].grid()
    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, 'test_signal')
    if show_plot:
        plt.show(block=True)


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
