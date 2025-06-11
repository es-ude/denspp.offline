import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.plot_helper import cm_to_inch, save_figure, scale_auto_value, get_plot_color, get_textsize_paper, get_plot_color_inactive


def plot_frames_feature(signals: dict, no_electrode: int, take_feat_dim: list=(0, 1),
                        path: str='', show_plot: bool=False) -> None:
    """Plotting the detected spike frame activity of used transient data
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param take_feat_dim:   List with dimension selection for plotting the 2d feature space
    :param path:            Path to save the figures
    :param show_plot:       If true, show plot
    :return:                None
    """
    assert len(take_feat_dim) == 2, "take_feat_dim must be 2 dimensional"
    frames_out = signals["frames"][0]
    cluster = signals["frames"][2]
    assert frames_out.shape[0] == cluster.size, "Dimensions between number of frames and corresponding cluster ID are not equal"
    feat = signals["features"]

    frames_mean = np.zeros(shape=(len(np.unique(cluster)), frames_out.shape[1]))
    for idx, id in enumerate(np.unique(cluster)):
        idx = np.argwhere(cluster == id).flatten()
        frames_mean[id, :] = np.mean(frames_out[idx], axis=0)

    plt.figure(figsize=(cm_to_inch(20), cm_to_inch(10)))
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133, sharex=ax1)

    ax1.set_title("Aligned Frames")
    ax1.plot(np.transpose(frames_out), marker='.', markersize=4, drawstyle='steps-post')

    ax2.set_title("Feature Space")
    for id in np.unique(cluster):
        idx = np.argwhere(cluster == id).flatten()
        ax2.plot(feat[idx, take_feat_dim[0]], feat[idx, take_feat_dim[1]],
                 color=get_plot_color(id), marker='.', linestyle='none')
    ax2.set_ylabel('Feat. 1')
    ax2.set_xlabel('Feat. 2')

    ax3.set_title("Mean Frames (Clustered)")
    for idx, frame in enumerate(frames_mean):
        ax3.plot(np.transpose(frame), color=get_plot_color(idx),
                 marker='.', markersize=4, drawstyle='steps-post')

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, f"pipeline_features_elec{str(no_electrode)}")
    if show_plot:
        plt.show(block=True)


def plot_transient_input_spikes(signals: dict, no_electrode: int, path: str= '', time_cut: list=(), show_plot: bool=False) -> None:
    """Plotting results of end-to-end signal processor with plotting the signal input and clustered spike events
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param path:            Path to save the figures
    :param time_cut:        Time cut
    :param show_plot:       If true, show plot
    :return:                None
    """
    # --- Selection of Transient signals
    fs_adc = signals["fs_dig"]
    xadc = signals["x_adc"]
    used_frames = signals["frames"]

    # --- Selection of FEC signals
    tD = np.arange(0, xadc.size, 1) / fs_adc
    frames_out = used_frames[0]
    ticks = used_frames[1]
    ticks_id = used_frames[2]
    cluster = np.unique(ticks_id)
    mean_frames = np.zeros(shape=(len(cluster), frames_out.shape[1]))
    for idx, id in enumerate(cluster):
        x0 = np.argwhere(ticks_id == id).flatten()
        mean_frames[idx, :] = np.mean(frames_out[x0], axis=0)

    # --- Plot 1: Transient signals
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(12)))
    plt.rcParams.update({'font.size': get_textsize_paper()})
    plt.subplots_adjust(hspace=0)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    ax1.plot(tD, xadc, color='k', drawstyle='steps-post')
    ax1.set_ylabel("ADC output")
    ax1.xaxis.set_visible(False)
    if not len(time_cut) == 0:
        ax1.set_xlim(time_cut)
    else:
        ax1.set_xlim([tD[0], tD[-1]])
    ax1.set_xlabel("Time t (s)")

    # Spike ticks
    for id in cluster:
        sel_x = np.where(ticks_id == id)[0]
        sel_ticks = ticks[sel_x]
        ax2.eventplot(positions=tD[sel_ticks], orientation="horizontal",
                      lineoffsets=0.45+id, linelengths=0.9,
                      color=get_plot_color(id))

    ax2.set_ylim([cluster[0], 1+cluster[-1]])
    ax2.set_ylabel("Spike Train")
    ax2.set_xlabel("Time t (s)")

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, f"pipeline_input_elec{str(no_electrode)}")
    if show_plot:
        plt.show(block=True)


def plot_transient_highlight_spikes(signals: dict, no_electrode: int,
                                    path: str="", time_cut: list=(), show_noise: bool=False, show_plot: bool=False) -> None:
    """Plotting the detected spike activity from transient data (highlighted, noise in gray)
    :param signals:         class containing the rawdata and processed data from class PipelineSignal
    :param no_electrode:    number of electrodes
    :param path:            Path to save the figures
    :param time_cut:        List for only specified range
    :param show_noise:      If true, show noise (otherwise flat line)
    :param show_plot:       If true, show plot
    :return:                None
    """
    fs_dig = signals["fs_dig"]
    xadc = signals["x_adc"]
    time = np.arange(0, xadc.size, 1) / fs_dig
    ticks = signals["frames"][1]
    ticks_id = signals["frames"][2]

    time0 = list()
    tran0 = list()
    colo0 = list()
    tick_old = 0
    for idx, tick in enumerate(ticks):
        sel = [int(tick)-12, int(tick)+30]
        time0.append(time[tick_old:sel[0]])
        time0.append(time[sel[0]:sel[1]])
        tran0.append(xadc[tick_old:sel[0]] if show_noise else np.zeros(shape=(len(xadc[tick_old:sel[0]]), ), dtype=int))
        tran0.append(xadc[sel[0]:sel[1]])
        colo0.append(get_plot_color_inactive())
        colo0.append(get_plot_color(ticks_id[idx]))
        tick_old = sel[1]

    # --- Plot generation
    plt.figure(figsize=(cm_to_inch(16), cm_to_inch(13)))
    # plt.subplots_adjust(hspace=0)
    axs = list()
    for idx in range(0, 1):
        axs.append(plt.subplot(1, 2, 1+2*idx))
        axs.append(plt.subplot(1, 2, 2+2*idx, sharey=axs[2*idx]))

    # Subplot 1: Transient signal (colored)
    for idx, time1 in enumerate(time0):
        axs[0].plot(time1, tran0[idx], linewidth=1, color=colo0[idx], drawstyle='steps-post')

    # --- Subplot 2: Histogram (from Subplot 1)
    no_bins = 1 + abs(max(xadc)) + abs(min(xadc))
    if not len(time_cut) == 0:
        sel0 = np.argwhere(time >= time_cut[0]).flatten()[0]
        sel1 = np.argwhere(time >= time_cut[1]).flatten()[0] -1
        x_bins = xadc[sel0:sel1]
    else:
        x_bins = xadc
    x_nonzero = np.where(x_bins != 0)[0]
    axs[1].hist(xadc[x_nonzero], color='k',
                density=True, log=True,
                bins=no_bins,
                orientation="horizontal")

    # --- Axis test
    axs[0].set_xlabel('Time t [s]')
    axs[0].set_ylabel('x_adc(t) [ ]')
    axs[0].grid()

    axs[1].set_xlabel('Density')
    axs[1].grid()

    # --- Zooming
    if not len(time_cut) == 0:
        axs[0].set_xlim(time_cut)
        addon_zoom = '_zoom'
    else:
        axs[0].set_xlim([time[0], time[-1]])
        addon_zoom = ''

    plt.tight_layout()
    # --- saving plots
    if path:
        save_figure(plt, path, f"pipeline_spikes_elec{str(no_electrode)}{addon_zoom}")
    if show_plot:
        plt.show(block=True)


def plot_mea_transient_total(mea_data: np.ndarray, mapping: np.ndarray, fs_used: float,
                             path2save: str='', do_global_limit: bool=False, do_show: bool=False) -> None:
    """Plotting the transient signals of the transient numpy signal with electrode information
    Args:
        mea_data:           Transient numpy array with neural signal [row, colomn, transient]
        mapping:            Numpy array with electrode mapping information
        fs_used:            Sampling rate of the signal [Hz]
        path2save:          Path for saving the figures
        do_global_limit:    Doing a global y-range setting
        do_show:            Show the plots
    Returns:
        None
    """
    assert mea_data.shape[0:2] == mapping.shape, "Shape mismatch, please apply_mapping() using PipelineCMDs"
    num_rows = mapping.shape[0]
    num_cols = mapping.shape[1]

    time_array = np.linspace(0, mea_data[0, 0].size, mea_data[0, 0].size) / fs_used
    scale_yaxis = scale_auto_value(mea_data)
    scale_xaxis = scale_auto_value(time_array)

    # Extract maximum values for scaling
    mea_yrange = np.zeros((np.sum(mapping > 0), 3), dtype=float)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if mapping[i, j] > 0:
                mea_yrange[idx, 0] = scale_yaxis[0] * np.min(mea_data[i, j])
                mea_yrange[idx, 1] = scale_yaxis[0] * np.max(mea_data[i, j])
                mea_yrange[idx, 2] = scale_yaxis[0] * (np.max(mea_data[i, j]) - np.min(mea_data[i, j]))
                idx += 1
    mea_yglobal = np.zeros((2, ), dtype=float)
    mea_yglobal[0] = np.min(mea_yrange[:, 0])
    mea_yglobal[1] = np.max(mea_yrange[:, 1])

    # --- Create the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, hspace=0.07, wspace=0.07)
    idx = 0
    ax_empty = list()
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            if mapping[i, j] < 1:
                ax.plot([0], 'k-', linewidth=0.1)
                ax_empty.append(ax)
            else:
                ax.plot(scale_xaxis[0] * time_array, scale_yaxis[0] * mea_data[i, j], 'k-', linewidth=1.0)
                ax.set_xlim([scale_xaxis[0] * time_array[0], scale_xaxis[0] * time_array[-1]])
                yrange_used = mea_yglobal.tolist() if do_global_limit else [mea_yrange[idx, 0], mea_yrange[idx, 1]]
                ax.set_ylim(yrange_used)
                idx += 1

            # Remove x-/y-axis ticks and labels
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_xticks([])
            # Remove subplot border
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

    if len(ax_empty):
        ax_empty[0].arrow(x=-0.35, y=0.5, dx=1, dy=0, length_includes_head=True, head_width=0.08, head_length=0.00002)
        ax_empty[0].text(x=0.15, y=0.55, s=f"{scale_xaxis[0] * time_array[-1]:.1f} {scale_xaxis[1]}s", ha='center')
        ax_empty[0].arrow(x=-0.35, y=0, dx=0, dy=1, length_includes_head=True, head_width=0.08, head_length=0.00002)
        ax_empty[0].text(x=-0.4, y=0.45, s=f"{scale_yaxis[0] * (mea_yglobal[1] - mea_yglobal[0]):.1f} {scale_yaxis[1]}V", ha='center', rotation=90)
    if path2save:
        save_figure(plt, path2save, 'mea_data' + ('_global' if do_global_limit else '_local'))
    if do_show:
        plt.show()
