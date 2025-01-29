import numpy as np
import matplotlib.pyplot as plt
from package.plot_helper import save_figure, scale_auto_value, get_plot_color
from package.metric.data import calculate_error_mse, calculate_error_rrmse
from package.data_process.transient_resampling import calculate_charge_injected


def plot_transient_stimulation(fs: float, voltage: np.ndarray, current: np.ndarray,
                               plot_charge: bool=False, take_range=(), zoom=(),
                               file_name: str='', path2save: str='', show_plot: bool=False) -> None:
    """Plotting the transient signal of stimulation trial
    Args:
        fs (float): Sampling rate of the transient signals
        voltage (np.ndarray): Transient voltage signal
        current (np.ndarray): Transient current signal
        plot_charge (bool, optional): Plot charge injected or not
        take_range (tuple, optional): Take range for plotting
        zoom (tuple, optional): Zoom range for plotting
        file_name (str, optional): File name to save figure
        path2save (str, optional): Path to save figure
        show_plot (bool, optional): Show plot
    Returns:
        None
    """
    # --- Definition of taking sample range for plotting
    time = np.linspace(0, voltage.size, voltage.size) / fs
    if len(take_range) == 2:
        x00 = int(np.argwhere(time >= take_range[0])[0])
        x01 = int(np.argwhere(time >= take_range[1])[0])
        time0 = time[x00:x01] - time[x00]
        voltage0 = voltage[x00:x01] - voltage[x00]
        current0 = current[x00:x01] - current[x00] + 102.5e-9
    else:
        time0 = time
        voltage0 = voltage
        current0 = current

    # --- Generating the plots
    fig, axs = plt.subplots(3 if plot_charge else 2, 1, sharex=True)
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": '.', "lines.markersize": 6})

    data_to_plot = [voltage0, current0]
    label_to_plot = [r"$\Delta U_{elec}$ [V]", r"$I_{elec}$ [A]"]
    if plot_charge:
        charge0 = calculate_charge_injected(current0, fs)
        data_to_plot.append(charge0)
        label_to_plot.append(r"$Q_{inj}$ [C]")

    xscale, xunit = scale_auto_value(time0)
    for idx, data in enumerate(data_to_plot):
        yscale, yunit = scale_auto_value(data)
        label = label_to_plot[idx][:-2] + yunit + label_to_plot[idx][-2:]
        axs[idx].plot(xscale * time0, yscale * data, color='k')
        axs[idx].set_ylabel(label)
        axs[idx].grid()

    axs[-1].set_xlabel(f"Time [{xunit}s]")
    axs[-1].set_xlim([xscale * time0[0], xscale * time0[-1]])

    # --- Zooming component
    if len(zoom) == 2:
        pos_zoom = [0.51, 0.05, 0.45, 0.6]
        x0 = int(np.argwhere(time0 >= zoom[0])[0])
        x1 = int(np.argwhere(time0 >= zoom[1])[0])
        for idx, data in enumerate(data_to_plot):
            yscale, yunit = scale_auto_value(data)
            axins0 = axs[idx].inset_axes(
                pos_zoom,
                xlim=(xscale * zoom[0], xscale * zoom[1]),
                ylim=(yscale * min(data[x0:x1]), yscale * max(data[x0:x1])),
                xticklabels=[], yticklabels=[]
            )
            axins0.plot(xscale * time0, yscale * data, color='k')
            axs[idx].indicate_inset_zoom(axins0, edgecolor="black")

    addon = '_zoomed' if len(zoom) == 2 else ''
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.0)

    file_name = file_name.split(".")[0] if file_name else 'stim_input'
    if path2save:
        save_figure(plt, path2save, f"{file_name}_transient{addon}")
    if show_plot:
        plt.show(block=True)


def plot_transient_fft(freq: np.ndarray, voltage: np.ndarray, current: np.ndarray,
                       file_name: str='', path2save: str='', show_plot: bool=False) -> None:
    """Plotting the FFT output of the transient signal
    :param freq:        Frequency array of spectral analysis
    :param voltage:     Numpy array with transient voltage signal
    :param current:     Numpy array with transient current signal
    :param file_name:   File name to save figure
    :param path2save:   Path to save figure
    :param show_plot:   Show plot
    :return:            None
    """
    plt.tick_params(direction='in')
    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": '.', "lines.markersize": 6})

    axs[0].loglog(freq, np.abs(voltage), color='k')
    axs[0].set_ylabel(r"fft($\Delta U_{elec}$)")

    axs[1].loglog(freq, np.abs(current), color='r')
    axs[1].set_xticks([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    axs[1].set_xlim([1e1, 1e6])
    axs[1].set_ylabel(r"fft($I_{elec}$)")
    axs[1].set_xlabel(r"Frequency f [Hz]")
    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    name = file_name.split(".")[0] if file_name else 'transient_signal'
    if path2save:
        save_figure(plt, path2save, f"{name}_fft_each")
    if show_plot:
        plt.show(block=True)


def plot_impedance(imp_stim=None, imp_eis=None, imp_fit=None, imp_mod=None,
                   name: str='', path2save: str='', show_plot: bool=False) -> None:
    """Plotting the impedance for different input modes (EIS, fitted and/or predicted)
    Args:
        imp_stim:   Dictionary with impedance and frequency values from transient stimulation signal fitting
        imp_eis:    Dictionary with impedance and frequency values from electrical impedance spectroscopy
        imp_fit:    Dictionary with impedance and frequency values from electrical impedance spectroscopy (fit)
        imp_mod:    Dictionary of impedance and frequency values from predicted model (using NGsolve)
        name:       Additional name for saving plot
        path2save:  Additional path for saving plot
        show_plot:  Saving and blocking plots
    Returns:
        None
    """
    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 14, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": '.', "lines.markersize": 6})

    data_complete = [imp_eis, imp_fit, imp_mod, imp_stim]
    label = ['EIS', 'EIS (Fit)', 'NGSolve', 'Transient', 'Min-Max']
    data_available = np.array([data is not None for data in data_complete])
    do_legend = np.sum(data_available) > 1

    idx = 0
    scale_imp = 1e-3
    # --- Data plotting
    xval_min = []
    xval_max = []
    for ite, data in enumerate(data_complete):
        if data is not None:
            xval_min.append(data['freq'].min())
            xval_max.append(data['freq'].max())
            if not (ite == 3 and len(data['Z'].shape) == 2):
                axs[0].plot(data['freq'], scale_imp * np.abs(data['Z']),
                            color=get_plot_color(idx), label=label[ite], marker='.')
                axs[1].plot(data['freq'], np.angle(data['Z'], deg=True),
                            color=get_plot_color(idx), label=label[ite], marker='.')
            else:
                # --- Plotting the amplitude
                ymean = scale_imp * np.abs(np.mean(data['Z'], axis=0))
                ymin = scale_imp * np.min(np.abs(data['Z']), axis=0)
                ymax = scale_imp * np.max(np.abs(data['Z']), axis=0)
                axs[0].plot(data['freq'], ymean, color=get_plot_color(idx), label=label[ite])
                axs[0].fill_between(data['freq'], ymean, y2=ymin, color='0.6', label=label[ite+1])
                axs[0].fill_between(data['freq'], ymean, y2=ymax, color='0.6')

                # --- Plotting the phase
                ymean = np.angle(np.mean(data['Z'], axis=0), deg=True)
                ymin = np.min(np.angle(data['Z'], deg=True), axis=0)
                ymax = np.max(np.angle(data['Z'], deg=True), axis=0)
                axs[1].plot(data['freq'], ymean, color=get_plot_color(idx), label=label[ite])
                axs[1].fill_between(data['freq'], ymean, y2=ymin, color='0.6', label=label[ite+1])
                axs[1].fill_between(data['freq'], ymean, y2=ymax, color='0.6')
            idx += 1
    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    # --- External stuff
    if do_legend:
        axs[0].legend()
    axs[0].set_ylabel(r'Impedance $|Z|$ [k$\Omega$]')
    axs[1].set_xlabel(r'Frequency $f$ [Hz]')
    axs[1].set_ylabel(r'Phase $\alpha$ [$\degree$]')
    axs[1].set_ylim([-90, 0])
    axs[1].set_yticks([-90, -75, -60, -45, -30, -15, 0])

    xlim_start = np.floor(np.log10(np.array(xval_min).min()))
    xlim_stop = np.ceil(np.log10(np.array(xval_max).min()))
    axs[1].set_xlim([10 ** xlim_start, 10 ** xlim_stop])
    for ax in axs:
        ax.grid()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)

    file_name = name.split(".")[0] if name else 'data'
    if path2save:
        save_figure(plt, path2save, f"{file_name}_imp_spectrum")
    if show_plot:
        plt.show(block=True)


def plot_impedance_error(freq: np.ndarray, imp_eis: np.ndarray, imp_fit: np.ndarray,
                         name: str="", path2save: str="", show_plot: bool=False) -> None:
    """Plotting the impedance error (fitted and measured)
    :param freq:        Frequency values of spectrum input
    :param imp_eis:     Impedance values from electrical impedance spectroscopy
    :param imp_fit:     Impedance values from ImpedanceFitting
    :param name:        File name of figure
    :param path2save:   Path for saving the data
    :param show_plot:   Showing and blocking plot
    :return:            None
    """
    imp_diff = np.abs(imp_eis - imp_fit) / np.abs(imp_eis)
    mse = calculate_error_mse(imp_eis, imp_fit)
    rrmse = calculate_error_rrmse(imp_eis, imp_fit)

    plt.figure()
    plt.title(f"MSE = {mse:.2f}, RRMSE = {rrmse:.2f}")
    plt.semilogx(freq, 100 * np.abs(imp_diff), color='k', label="Rel. Delta")
    plt.grid()
    plt.ylabel(r'Rel. $\Delta$ [%]')
    plt.xlabel(r'Frequency [Hz]')

    xlim_start = np.floor(np.log10(freq[0]))
    xlim_stop = np.ceil(np.log10(freq[-1]))
    plt.xlim([10 ** xlim_start, 10 ** xlim_stop])
    plt.tight_layout()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": '.', "lines.markersize": 6})

    file_name = name.split(".")[0]
    if path2save:
        save_figure(plt, path2save, f"{file_name}_Z_error")
    if show_plot:
        plt.show(block=True)
