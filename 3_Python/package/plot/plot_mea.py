import numpy as np
import matplotlib.pyplot as plt
import mplcursors
import pickle
from os import mkdir
from os.path import exists, join

from package.plot.plot_common import save_figure, cm_to_inch, scale_auto_value
from package.data_call.call_handler import DataHandler


def results_mea_transient_total(mea_data: np.ndarray | list, config: DataHandler, path2save="",
                                yscale=1e6, do_global_limit=False) -> None:
    """Plotting the transient signals of the transient numpy signal with electrode information
    Args:
        mea_data:           Transient numpy array with neural signal [row, colomn, transient]
        config:             DataHandler which contains mapping informtion
        path2save:          Path for saving the figures
    Returns:
        None
    """
    # --- Generating folder if not exists
    if path2save:
        if not exists(path2save):
            mkdir(path2save)

    # --- Preparing the figure
    plotted_lines = []
    num_rows = config.mapping_dimension[0]
    num_cols = config.mapping_dimension[1]

    time_array = np.linspace(0, mea_data[0, 0].size, mea_data[0, 0].size) / config.data_fs_used
    scale_xaxis = scale_auto_value(time_array)

    # Extract maximum values for scaling
    mea_yrange = np.zeros((np.sum(config.mapping_active == True), 3), dtype=float)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if config.mapping_active[i, j]:
                mea_yrange[idx, 0] = yscale * np.min(mea_data[i, j])
                mea_yrange[idx, 1] = yscale * np.max(mea_data[i, j])
                mea_yrange[idx, 2] = yscale * (np.max(mea_data[i, j]) - np.min(mea_data[i, j]))
                idx += 1
    mea_yglobal = np.zeros((2, ), dtype=float)
    mea_yglobal[0] = np.min(mea_yrange[:, 0])
    mea_yglobal[1] = np.max(mea_yrange[:, 1])

    # --- Create the figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, hspace=0.07, wspace=0.07)
    idx = 0
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i, j]
            if config.mapping_active[i, j]:
                line, = ax.plot(scale_xaxis[0] * time_array, yscale * mea_data[i, j], 'k-', linewidth=1.0)
                ax.set_xlim([scale_xaxis[0] * time_array[0], scale_xaxis[0] * time_array[-1]])
                yrange_used = mea_yglobal.tolist() if do_global_limit else [mea_yrange[idx, 0], mea_yrange[idx, 1]]
                ax.set_ylim(yrange_used)

                if not do_global_limit:
                    ax.arrow(x=0, y=0, dx=0, dy=1)
                    ax.text(x=-0.25, y=0.5, s=f"{yrange_used[1]-yrange_used[0]:.1f} µV", ha='center', rotation=90)
                idx += 1
            else:
                line, = ax.plot([0], 'k-', linewidth=0.1)

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

            # Store the subplot and the data for the cursor event
            if config.mapping_active[i, j]:
                line._mea_data = mea_data[i, j]
            else:
                line._mea_data = [0]
            plotted_lines.append(line)

    # Add meta information
    plt.suptitle(f'Neural Activities of {config.data_type} [µV]', y=1)

    # Add arrows
    plt.arrow(x=0, y=0, dx=1, dy=0)
    plt.text(x=0.5, y=-0.25, s=f"{scale_xaxis[0] * time_array[-1]} {scale_xaxis[1]}s", ha='center')
    if do_global_limit:
        plt.arrow(x=0.5, y=0, dx=0, dy=1)
        plt.text(x=0.35, y=0.5, s=f"{mea_yglobal[1]-mea_yglobal[0]:.0f} µV", ha='center', rotation=90)

    # Plotting and saving
    if path2save:
        save_figure(plt, path2save, 'mea_data' + ('_global' if do_global_limit else '_local'))
    plt.show()


def results_mea_transient_single(mea_data: np.ndarray | list, config: DataHandler, path2save="") -> None:
    """Plotting the transient signals of the transient numpy signal with electrode information
    Args:
        mea_data:           Transient numpy array with neural signal [row, colomn, transient]
        config:             DataHandler which contains mapping informtion
        path2save:          Path for saving the figures
    Returns:
        None
    """
    # Function to create a bigger plot when clicking on a subplot
    def on_click(event):
        artist = event.artist
        data = artist._mea_data
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(data, 'bo-')
        ax.set_xlabel('Time')
        ax.set_ylabel('Voltage')
        ax.set_title('Channel')
        plt.show()

    # Use mplcursors to enable click events
    cursor = mplcursors.cursor(plotted_lines, hover=False)
    cursor.connect("add", on_click)

    with open(join(path2save, 'Plotted_Signals.fig.pickle'), 'wb') as f:
        pickle.dump(fig, f)

    # --- Generating subplots of each channel
    for i in range(num_rows):
        for j in range(num_cols):
            if config.mapping_active[i, j]:
                scale_yaxis = scale_auto_value(mea_data[i, j])
                plt.plot(scale_xaxis[0] * time_array, scale_yaxis[0] * mea_data[i, j], 'k', linewidth=1.0)
                plt.ylabel(f"Voltage Signal [{scale_yaxis[1]}V]")
                plt.xlabel(f"Time [{scale_xaxis[1]}s]")
                plt.xlim([scale_xaxis[0] * time_array[0], scale_xaxis[0] * time_array[-1]])
                plt.grid()
                plt.tight_layout()

                if path2save:
                    save_figure(plt, path2save, f'channel_{str(config.mapping_used[i][j])}')
                plt.close()
