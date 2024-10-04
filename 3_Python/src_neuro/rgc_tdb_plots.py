import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from package.plot.plot_common import save_figure, cm_to_inch, scale_auto_value
from package.data_call.call_spike_files import DataLoader, SettingsDATA
from package.data_call.call_cellbib import CellSelector
from src_neuro.pipeline_data import Pipeline


data_set = SettingsDATA(
    path="/home/erbsloeh",
    data_set=7, data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
    fs_resample=40e3
)


def plot_rgc_transient_signals(rawdata: list, frames: list, cell_name: list,
                               fs: float, path2save='', show_plots=False) -> None:
    """Plotting an example of transient signals and spike frames from the RGC TDB
    Args:
        rawdata:    List with transient rawdata from ADC conversion
        frames:     List with extracted frames from the spike detection
        cell_name:  List with cell name of the corresponding recording
        fs:         Sampling rate
        path2save:  Information for saving the figures (if empty, no saving)
        show_plots: Boolean for showing and blocking the plots
    Returns:
         None
    """
    # --- Plotting the results
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(cm_to_inch(14), cm_to_inch(16)))
    tick_textsize = 12
    label_fontsize = 13

    scaley, unity = scale_auto_value(rawdata[0])
    for idx, data in enumerate(rawdata):
        time = np.arange(0, data.size) / fs
        scalex, unitx = scale_auto_value(time)
        axs[idx, 0].plot(scalex * time, scaley * data, color='k', linewidth=1)
        axs[idx, 0].set_ylabel(f'ADC output [{unity}V]', fontsize=label_fontsize)
        axs[idx, 0].grid()
        axs[idx, 0].tick_params(direction='out', labelsize=tick_textsize)
        axs[idx, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[idx, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].set_xlim([2.3, 3.5])
    axs[1, 0].set_xticks([2.3, 2.6, 2.9, 3.2, 3.5])
    axs[1, 0].set_xlabel(f'Time [{unity}s]', fontsize=label_fontsize)
    axs[0, 0].set_title(cell_name[0], loc='left', fontdict={'fontsize': tick_textsize})
    axs[1, 0].set_title(cell_name[1], loc='left', fontdict={'fontsize': tick_textsize})

    # Plot spike shape
    for idx, frame_ch in enumerate(frames):
        mean_frame = np.median(frame_ch, axis=0)
        axs[idx, 1].plot(scaley * np.transpose(frame_ch), linewidth=0.5)
        axs[idx, 1].plot(scaley * mean_frame, linewidth=2, color='k')
        axs[idx, 1].grid()
        axs[idx, 1].tick_params(direction='out', labelsize=tick_textsize)
        axs[idx, 1].xaxis.set_minor_locator(AutoMinorLocator())
        axs[idx, 1].yaxis.set_minor_locator(AutoMinorLocator())
    axs[0, 1].set_xlim([0, 31])
    axs[0, 1].set_xticks([0, 7, 15, 23, 31])
    axs[1, 1].set_xlabel('Frame position', fontsize=label_fontsize)

    # Setting Ticks and Limits
    axs[0, 0].set_ylim([-70, 70])
    axs[0, 0].set_yticks([-70, -35, 0, 35, 70])
    axs[1, 0].set_ylim([-160, 120])
    axs[1, 0].set_yticks([-160, -80, 0, 60, 120])

    # Output
    plt.tight_layout(h_pad=0.05, w_pad=0.05)
    if path2save:
        save_figure(fig, path2save, 'rgc_tdb_transient_example')
    if show_plots:
        plt.show(block=True)


if __name__ == "__main__":
    file_numbers = [19, 36]
    channel_number = [611, 130]

    # --- Loading the src_neuro
    cell_bib = CellSelector(1, 0)

    # --- Start Routine
    print("\nGetting the given data points from the RGC TDB for plotting")
    print(f"... loading the datasets")
    rawdata0 = list()
    frames0 = list()
    cell_name0 = list()
    adc_fs = 0.0
    for idx, file in enumerate(file_numbers):
        ch = channel_number[idx]

        # --- Getting the Data
        data_set.data_point = file
        datahandler = DataLoader(data_set)
        datahandler.do_call()
        datahandler.do_resample()

        # --- Processing the analogue input (channel specific)
        afe = Pipeline(data_set.fs_resample)

        spike_xpos = np.floor(datahandler.raw_data.evnt_xpos[ch] * afe.fs_adc / afe.fs_ana).astype("int")
        afe.run_input(datahandler.raw_data.data_raw[ch], spike_xpos)
        adc_lsb = afe.lsb
        adc_fs = afe.fs_adc

        # --- Getting the results
        rawdata0.append(adc_lsb * afe.signals.x_adc)
        frames0.append(adc_lsb * afe.signals.frames_align)
        for id in np.unique(datahandler.raw_data.evnt_id[ch]):
            cell_name0.append(cell_bib.get_celltype_name_from_id(int(id)))

    # Delete after runs
    del spike_xpos, datahandler, idx, ch, file

    print('... plot the results!')
    # Plot transient signals
    plot_rgc_transient_signals(rawdata0, frames0, cell_name0, adc_fs, path2save='runs', show_plots=True)
