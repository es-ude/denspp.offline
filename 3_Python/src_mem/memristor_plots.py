import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from package.plot.plot_common import cm_to_inch, save_figure
from src_mem.pipeline_mem import PipelineSignal


def show_plots(block=True) -> None:
    """Showing plots and blocking system if required"""
    plt.show(block=block)


def plt_memristor_ref(frames_in: np.ndarray, frames_cluster: np.ndarray, frames_mean: np.ndarray) -> None:
    """Plotting reference signals for testing with BFO memristor-based calculation"""
    color = ['k', 'r', 'b', 'g', 'y', 'c', 'm']
    path = 'runs'
    fig_size = [22, 8]
    textsize = 12
    yrange = [-20, 60]
    error = [-70.3, -111.1, 0.2, 72.85, -119.5]

    use_class = 2
    sel_pos = np.where(frames_cluster == use_class)
    frames_input = frames_in[sel_pos[0], :]

    # --- Plot #1
    plt.figure()
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(cm_to_inch(fig_size[0]), cm_to_inch(fig_size[1]-2)))
    plt.subplots_adjust(hspace=0, wspace=0.5)

    ax = [plt.subplot(1, 5, i+1) for i in range(5)]
    ax[0].set_ylabel("U_top (Sample)")
    for a in ax:
        a.set_ylim(yrange)
        a.plot(np.transpose(frames_input), linewidth=1)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(direction='in')
        a.grid()

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if path:
        save_figure(plt, path, "memristor_input")

    # --- Plot #2
    plt.figure()
    plt.rcParams.update({'font.size': textsize})
    plt.figure(figsize=(cm_to_inch(fig_size[0]), cm_to_inch(fig_size[1])))
    plt.subplots_adjust(hspace=0, wspace=0.5)

    ax = [plt.subplot(1, 5, i + 1) for i in range(5)]
    ax[0].set_ylabel("U_bot (Ref.)")
    for idx, a in enumerate(ax):
        a.set_ylim(yrange)
        a.plot(frames_mean[idx, :], color=color[idx], linewidth=2)
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.tick_params(direction='in')
        a.grid()
        a.set_title(f'Class {idx}\nDelta = {error[idx]:.1f}')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    if path:
        save_figure(plt, path, "memristor_reference")


def _plt_pipeline_signals(plot_signals: dict, fs_sig: float, ite_num=-1, path2save='', addon_name='') -> None:
    """"""
    # --- Plotting transient results
    plt.figure()
    axs = [plt.subplot(len(plot_signals), 1, idx + 1) for idx in range(0, len(plot_signals))]

    for idx, signal in enumerate(plot_signals.items()):
        signal2plot = signal[1] if signal[0] == "Input Signal" else signal[1][1,]
        time0 = np.linspace(0, signal2plot.size, signal2plot.size) / fs_sig

        axs[idx].plot(time0, signal2plot, color='k', label=signal[0])

        axs[idx].set_ylabel(r'Signal $f_x(t)$')
        axs[idx].set_xlim(time0[0], time0[-1])
        axs[idx].legend()
        axs[idx].grid()

    axs[-1].set_xlabel(r'Time $t$ / s')
    plt.tight_layout()
    if path2save:
        addon = "" if ite_num == -1 else f"_run{ite_num:03d}"
        save_figure(plt, path2save, "pipeline_mem_signals" + addon_name + addon)


def plt_pipeline_signals_part_one(signals: PipelineSignal, idx=-1, path2save='') -> None:
    """"""
    fs_sig = signals.fs_ana
    plot_signals = dict()
    plot_signals.update({"Input Signal": signals.u_inp})
    plot_signals.update({"Device Top": signals.u_mem_top})
    plot_signals.update({"Device Bottom": signals.u_mem_bot})

    _plt_pipeline_signals(plot_signals, fs_sig, idx, path2save, '_part_one')


def plt_pipeline_signals_part_two(signals: PipelineSignal, idx=-1, path2save='') -> None:
    """"""
    fs_sig = signals.fs_ana
    plot_signals = dict()
    plot_signals.update({"Current Diff. Output": signals.i_tra})
    plot_signals.update({"Transimpedance Output": signals.u_tra})
    plot_signals.update({"Integrator Output": signals.u_int})

    _plt_pipeline_signals(plot_signals, fs_sig, idx, path2save, '_part_two')


def plot_pipeline_feat(feat_array: np.ndarray, label=None, dict=None, path2save='') -> None:
    """Plotting the feature space"""
    color = 'krbgycm'
    num_dim = feat_array.shape[1]

    # --- Processing data
    is_label_available = label is not None
    if not is_label_available:
        data_dim = [list() for idx in range(0, num_dim)]
        num_label = 0
        for idx_dim, data_empty in enumerate(data_dim):
            data_dim[idx_dim] = feat_array[:, idx_dim]
    else:
        data_dim = [list() for idx in range(0, num_dim)]
        for idx_dim, data_empty in enumerate(data_dim):
            data_label = list()
            for num_label in np.unique(label):
                pos = np.argwhere(label == num_label)
                data_label.append(feat_array[pos, idx_dim])
            data_dim[idx_dim] = data_label
        num_label = np.unique(label).size - 1

    # --- Plotting
    if num_dim > 3:
        print("Feature space can not be plotted due to higher dimension than 3!")
    else:
        fig = plt.figure()
        if num_dim == 2:
            # Run plotting for 2D
            if not is_label_available:
                plt.plot(data_dim[0], data_dim[1], marker='.', linestyle='None')
            else:
                for i in range(0, num_label+1):
                    if dict is None:
                        plt.plot(data_dim[0][i], data_dim[1][i],
                                 marker='.', linestyle='None', color=color[i % 7])
                    else:
                        plt.plot(data_dim[0][i], data_dim[1][i],
                                 marker='.', linestyle='None', color=color[i % 7], label=dict[i])
                        plt.legend()
            plt.xlabel('Feat[0]')
            plt.ylabel('Feat[1]')
        else:
            # Run plotting for 3D
            Axes3D(fig)
            ax = plt.axes(projection='3d')
            for i in range(0, num_label+1):
                if dict is None:
                    ax.scatter3D(data_dim[0][i], data_dim[1][i], data_dim[2][i], color=color[i % 7], marker='.')
                else:
                    ax.scatter3D(data_dim[0][i], data_dim[1][i], data_dim[2][i], color=color[i % 7], marker='.', label=dict[i])
            ax.set_xlabel('Feat[0]')
            ax.set_ylabel('Feat[1]')
            ax.set_zlabel('Feat[2]')
            ax.legend()

        plt.grid()
        plt.tight_layout()
        if path2save:
            save_figure(plt, path2save, "pipeline_mem_feat")
