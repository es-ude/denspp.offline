import numpy as np
import matplotlib.pyplot as plt
from package.plot.helper import save_figure, scale_auto_value, _get_median


def plot_hist_impedance(r_tis: np.ndarray, z_war: np.ndarray, c_dl: np.ndarray, r_ct: np.ndarray,
                        name='', path2save='', show_plot=False) -> None:
    """Plotting the Histogram of the Parameters of Impedance Model"""
    plt.clf()
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": 'None', "lines.markersize": 0})

    ax = [plt.subplot(1, 4, i + 1) for i in range(4)]

    ax[0].hist(1e9 * c_dl)
    ax[0].set_xlabel(r"$C_{dl}$ [nF]")
    ax[0].grid()

    ax[1].hist(1e-9 * r_ct)
    ax[1].set_xlabel(r"$R_{ct}$ [M$\Omega$]")
    ax[1].grid()

    ax[2].hist(1e-6 * z_war)
    ax[2].set_xlabel(r"$Z_{war}$ [M$\Omega$/sqrt(Hz)]")
    ax[2].grid()

    ax[3].hist(1e-3 * r_tis)
    ax[3].set_xlabel(r"$R_{tis}$ [k$\Omega$]")
    ax[3].grid()

    plt.tight_layout()
    if path2save:
        save_figure(path2save, f"{name}_hist")
    if show_plot:
        plt.show(block=True)


def plot_sweep_params(x0: np.ndarray, type_in: str, r_tis: np.ndarray, z_war: np.ndarray,
                      c_dl: np.ndarray, r_ct: np.ndarray, zoom=(),
                      name='', path2save='', show_plot=False) -> None:
    """Plotting the sweeping results with the electrical target parameter
    Args:
        x0:         Input of the parameter sweep
        type_in:    Type of the parameter sweep ['LSB', 'fs']
        r_tis:      Array with parameters of Tissue Resistance
        z_war:      Array with parameters of Warburg Impedance
        c_dl:       Array with parameters of Faraday Capacity
        r_ct:       Array with parameters of Faraday Resistance
        zoom:       Zooming area
        name:       Title of the Plot
        path2save:  Path to save the figure
        show_plot:
    Returns:
        None
    """
    if isinstance(zoom, list):
        if len(zoom) == 2:
            fs0 = x0[zoom[0]:zoom[1]]
            rs0 = r_tis[zoom[0]:zoom[1]]
            rc0 = r_ct[zoom[0]:zoom[1]]
            cd0 = c_dl[zoom[0]:zoom[1]]
            zw0 = z_war[zoom[0]:zoom[1]]
            addon = '_zoom'
        else:
            fs0 = x0
            rs0 = r_tis
            rc0 = r_ct
            cd0 = c_dl
            zw0 = z_war
            addon = ''
    else:
        fs0 = x0
        rs0 = r_tis
        rc0 = r_ct
        cd0 = c_dl
        zw0 = z_war
        addon = ''

    # Type definition
    if type_in == 'LSB':
        addon_type = '_lsb'
        xscale = 1e3
    elif type_in == 'fs':
        addon_type = '_fs'
        xscale = 1
    else:
        addon_type = '_false'
        xscale = 1

    plt.clf()
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": 'None', "lines.markersize": 12})

    axs = list()
    for idx in range(0, 4):
        axs.append(plt.subplot(4, 1, 1+idx))

    scale = 1e-3
    axs[0].semilogx(fs0 * xscale, scale * rs0[:, 1], color='k', marker='.')
    axs[0].errorbar(fs0 * xscale, scale * rs0[:, 1], yerr=scale * rs0[:, 3], color='k')
    axs[0].fill_between(fs0 * xscale, scale * rs0[:, 0], scale * rs0[:, 2], lw=1, alpha=0.4, facecolor='green')
    axs[0].set_ylabel(r'$R_{tis}$ [k$\Omega$]')
    axs[0].set_xticks([])

    scale = 1e-6
    axs[1].semilogx(fs0 * xscale, scale * zw0[:, 1], color='k', marker='.')
    axs[1].errorbar(fs0 * xscale, scale * zw0[:, 1], yerr=scale * zw0[:, 3], color='k')
    axs[1].fill_between(fs0 * xscale, scale * zw0[:, 0], scale * zw0[:, 2], lw=1, alpha=0.4, facecolor='green')
    axs[1].set_ylabel(r'$Z_{war}$ [M$\Omega$/$\sqrt{Hz}$]')
    axs[1].set_xticks([])

    scale = 1e9
    axs[2].semilogx(fs0 * xscale, scale * cd0[:, 1], color='k', marker='.')
    axs[2].errorbar(fs0 * xscale, scale * cd0[:, 1], yerr=scale * cd0[:, 3], color='k')
    axs[2].fill_between(fs0 * xscale, scale * cd0[:, 0], scale * cd0[:, 2], lw=1, alpha=0.4, facecolor='green')
    axs[2].set_ylabel(r'$C_{dl}$ [nF]')
    axs[2].set_xticks([])

    scale = 1e-9
    axs[3].semilogx(fs0 * xscale, scale * rc0[:, 1], color='k', marker='.')
    axs[3].errorbar(fs0 * xscale, scale * rc0[:, 1], yerr=scale * rc0[:, 3], color='k')
    axs[3].fill_between(fs0 * xscale, scale * rc0[:, 0], scale * rc0[:, 2], lw=1, alpha=0.4, facecolor='green')
    axs[3].set_ylabel(r"$R_{ct}$ [G$\Omega$]")
    if type_in == 'LSB':
        axs[3].set_xlabel(r'Least Significant Bit [mV]')
    else:
        axs[3].set_xlabel(r'Sampling frequency [Hz]')

    for ax in axs:
        ax.grid()
        ax.locator_params(axis='y', nbins=4)

    plt.tight_layout(pad=0.2)
    if path2save:
        save_figure(path2save, f"{name}_sweep_params{addon_type}{addon}")
    if show_plot:
        plt.show(block=True)


def plot_sweep_metric(x0: np.ndarray, type_in: str, metric: np.ndarray, type_name: str,
                      name='', path2save='', show_plot=False) -> None:
    """Plotting one metric of the sweep run

    Args:
        x0: Input of the parameter sweep
        type_in: Type of the parameter sweep ['LSB', 'fs']
        metric: Metric results
    """
    # Type definition
    if type_in == 'LSB':
        addon_type = '_lsb'
        xscale = 1e3
    elif type_in == 'fs':
        addon_type = '_fs'
        xscale = 1
    else:
        addon_type = '_false'
        xscale = 1

    plt.clf()
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, 'lines.marker': '.', 'lines.markersize': 12})

    plt.title(f"{type_name} = {np.mean(metric[:, 1]):.4f}")
    plt.semilogx(x0 * xscale, metric[:, 1], color='k', label='mean')
    plt.errorbar(x0 * xscale, metric[:, 1], yerr=metric[:, 3], color='k')
    plt.fill_between(x0 * xscale, metric[:, 0], metric[:, 2], lw=1, alpha=0.4, facecolor='green')
    if type_in == 'LSB':
        plt.xlabel(r'Least Significant Bit [mV]')
    else:
        plt.xlabel(r'Sampling frequency [Hz]')

    plt.ylabel(f"{type_name}")
    plt.grid()

    plt.tight_layout(pad=0.2)
    if path2save:
        save_figure(path2save, f"{name}_metric{addon_type}_{type_name}")
    if show_plot:
        plt.show(block=True)


def plot_boxplot_params(x0: np.ndarray, type_in: str, r_tis: list, z_war: list, c_dl: list, r_ct: list,
                        given_ticks=True, plot_cdl=True, plot_rtis=True, plot_rct=False, plot_zwar=True,
                        name='', path2save='', show_plot=False) -> None:
    """Plotting the results with boxplot

    Args:
        x0:         Input of the parameter sweep
        type_in:    Type of the parameter sweep ['LSB', 'fs']
        r_tis:      List of tissue resistance value
        z_war:      List of
        c_dl:       List of
        r_ct:       List of
        given_ticks: Bool for taking the pre-defined ticks
        plot_cdl:   Do plotting the Cdl output
        plot_rtis:  Do plotting the
        plot_rct:   Do plotting the
        plot_zwar:  Do plotting the
        name:       Name for saving the plots
        path2save:  Path for saving the plots
    """
    # Type definition
    if type_in == 'LSB':
        addon_type = '_lsb'
        xscale = 1e3
    elif type_in == 'fs':
        addon_type = '_fs'
        xscale = 1
    else:
        addon_type = '_false'
        xscale = 1

    # --- Pre-Processing
    w = 0.06
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
    scale = lambda p, w: [val * w for val in p]

    # --- Plot #1
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, "lines.marker": 'None', "lines.markersize": 12})

    axs = list()
    row_num = sum(np.array((plot_cdl, plot_rtis, plot_zwar, plot_rct)) == True) + 1
    row_idx = 0
    for idx in range(0, row_num):
        axs.append(plt.subplot(row_num, 1, 1+idx))

    if plot_rtis:
        axs[row_idx].boxplot(scale(r_tis, 1e-3), positions=xscale*x0, widths=width(xscale*x0, w),
                             patch_artist=True, showfliers=False)
        axs[row_idx].set_ylabel(r'$R_\mathrm{tis}$ [k$\Omega$]')
        if given_ticks:
            axs[row_idx].set_ylim([8.5, 15.5])
            axs[row_idx].set_yticks([9, 11, 13, 15])
        row_idx += 1
    if plot_zwar:
        axs[row_idx].boxplot(scale(z_war, 1e-6), positions=xscale*x0, widths=width(xscale*x0, w),
                                   patch_artist=True, showfliers=False)
        axs[row_idx].set_ylabel(r'$Z_\mathrm{war}$' + '\n' + r'[M$\Omega$/$\sqrt{Hz}$]')
        if given_ticks:
            axs[row_idx].set_ylim([0.1, 2.65])
            axs[row_idx].set_yticks([0.25, 1.0, 1.75, 2.5])
        row_idx += 1
    if plot_cdl:
        axs[row_idx].boxplot(scale(c_dl, 1e9), positions=xscale * x0, widths=width(xscale * x0, w),
                                   patch_artist=True, showfliers=False)
        axs[row_idx].set_ylabel(r'$C_\mathrm{dl}$ [nF]')
        if given_ticks:
            axs[row_idx].set_ylim([0, 850])
            axs[row_idx].set_yticks([50, 300, 550, 800])
        row_idx += 1
        # --- Adding a zoom
        axs[row_idx].boxplot(scale(c_dl, 1e9), positions=xscale * x0, widths=width(xscale * x0, w),
                             patch_artist=True, showfliers=False)
        axs[row_idx].set_ylabel(r'$C_\mathrm{dl}$ [nF]')
        if given_ticks:
            axs[row_idx].set_ylim([30, 140])
            axs[row_idx].set_yticks([40, 70, 100, 130])
        row_idx += 1
    if plot_rct:
        axs[row_idx].boxplot(scale(r_ct, 1e-6), positions=xscale * x0, widths=width(xscale * x0, w),
                                   patch_artist=True, showfliers=False)
        axs[row_idx].set_ylabel(r"$R_\mathrm{ct}$ [M$\Omega$]")
        if given_ticks:
            axs[row_idx].set_ylim([0.5, 4.5])
            axs[row_idx].set_yticks([1, 2, 3, 4])

    if type_in == 'LSB':
        plt.xlabel(r'Least Significant Bit [mV]')
    else:
        plt.xlabel(r'Sampling frequency [Hz]')

    for ax in axs:
        ax.grid()
        ax.set_xscale('log')
        ax.locator_params(axis='y', nbins=4)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(hspace=0.08, wspace=0.0)

    if path2save:
        save_figure(path2save, f"{name}_sweep_params{addon_type}")
    if show_plot:
        plt.show(block=True)


def plot_boxplot_metric(x0: np.ndarray, type_in: str, metric: list, type_name: str,
                        name='', path2save='', show_plot=False) -> None:
    """Plotting one metric of the sweep run

    Args:
        x0: Input of the parameter sweep
        type_in: Type of the parameter sweep ['LSB', 'fs']
        metric: Metric results
    """
    # Type definition
    if type_in == 'LSB':
        addon_type = '_lsb'
        xscale = 1e3
    elif type_in == 'fs':
        addon_type = '_fs'
        xscale = 1
    else:
        addon_type = '_false'
        xscale = 1

    # --- Pre-Processing
    w = 0.1
    width = lambda p, w: 10 ** (np.log10(p) + w / 2.) - 10 ** (np.log10(p) - w / 2.)
    mean_metric = _get_median(metric)

    plt.clf()
    plt.figure()
    plt.rcParams.clear()
    plt.rcParams.update({'font.size': 13, 'font.family': 'serif',
                         "lines.linewidth": 1, 'lines.marker': '.', 'lines.markersize': 12})

    plt.title(f"{type_name} = {mean_metric:.4f}")
    plt.boxplot(metric, positions=xscale*x0, widths=width(xscale*x0, w),
                patch_artist=True, showfliers=False)
    plt.xscale('log')
    if type_in == 'LSB':
        plt.xlabel(r'Least Significant Bit [mV]')
    else:
        plt.xlabel(r'Sampling frequency [Hz]')
    plt.ylabel(f"{type_name}")
    plt.grid()

    plt.tight_layout(pad=0.2)
    if path2save:
        save_figure(path2save, f"{name}_metric-box{addon_type}_{type_name}")
    if show_plot:
        plt.show(block=True)


def plot_heatmap_2d_metric(freq: np.ndarray, lsb: np.ndarray, metric: list,
                           type_metric: str, mdict_eis=(),
                           name='', path2save='', show_plot=False) -> None:
    """Plotting the heatmap with the median metric from the 2D-Parameter Sweep"""
    # --- Pre-Processing
    num_fs = np.unique(freq).size
    num_lsb = np.unique(lsb).size

    title_addon = '' if not len(mdict_eis) else r'$\Delta$'

    title_text = type_metric
    text_size = 9
    match type_metric:
        case 'Rtis':
            title_text = title_addon + r'$R_\mathrm{tis}$'
            text_size = 10
        case 'Cdl':
            title_text = title_addon + r'$C_\mathrm{dl}$'
            text_size = 9
        case 'Zw':
            title_text = title_addon + r'$Z_\mathrm{w}$'
            text_size = 9
        case 'Rct':
            title_text = title_addon + r'$R_\mathrm{ct}$'
            text_size = 9
        case 'MAPE':
            title_text = type_metric
            text_size = 11

    # --- Getting the data for plot
    X, Y = np.meshgrid(np.unique(np.log10(freq)), np.unique(np.log10(lsb)))
    Z = np.zeros(shape=X.shape, dtype=float)
    for idx, val in enumerate(metric):
        used_fs = np.log10(freq)[idx]
        used_lsb = np.log10(lsb)[idx]
        sel_col = int(np.argwhere(used_fs == np.log10(freq)).flatten()[0] / num_lsb)
        sel_row = int(np.argwhere(used_lsb == np.log10(lsb)).flatten()[0])

        if val.size == 0:
            Z[sel_row, sel_col] = 0
        else:
            Z[sel_row, sel_col] = np.median(np.array(val, dtype=float))
    del sel_row, sel_col, num_fs, num_lsb

    # --- MAPE calculation with given value
    if not len(mdict_eis) == 0:
        if type_metric in mdict_eis:
            Z = np.abs(Z - mdict_eis[type_metric])

    # --- Plotting
    plt.figure(figsize=(12, 9))
    plt.rcParams.update({'font.size': 20})

    plt.pcolor(X, Y, Z, edgecolor='k', vmin=Z.min(), vmax=Z.max())
    plt.xlabel('Sampling Frequency [log10(Hz)]')
    plt.ylabel('Least Significant Bit [log10(V)]')
    plt.title(title_text)

    # --- Add text
    for (j, i), label in np.ndenumerate(Z):
        plt.text(X[j, i], Y[j, i], f'{label:.2f}', ha='center', va='center',
                 fontdict={'size': text_size}, color='gainsboro')

    plt.colorbar(shrink=0.95, aspect=25)
    plt.tight_layout()

    # --- Saving
    if path2save:
        save_figure(plt, path2save, f"{name}_metric-box2d_{type_metric}")
    if show_plot:
        plt.show(block=True)
