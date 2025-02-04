import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.signal import find_peaks

from denspp.offline.structure_builder import get_path_project_start
from denspp.offline.plot_helper import save_figure, cm_to_inch, scale_auto_value
from denspp.offline.data_call.call_cellbib import CellSelector

from src_neuro.call_spike import DataLoader, SettingsDATA
from denspp.offline.data_merge.pipeline_data import Pipeline


def plot_rgc_transient_signals(rawdata: list, frames: list, spike_xpos: list, cell_name: list,
                               fs: float, path2save='', show_plots=False) -> None:
    """Plotting an example of transient signals and spike frames from the RGC TDB
    Args:
        rawdata:    List with transient rawdata from ADC conversion
        frames:     List with extracted frames from the spike detection
        spike_xpos: Numpy array with position of spikes
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
    xpos_lim = [2.3, 3.5]
    xpos_start = int(xpos_lim[0] * fs)
    xpos_end = int(xpos_lim[1] * fs)

    scaley, unity = (1e4, 'µ') # scale_auto_value(rawdata[0][xpos_start:xpos_end+1])
    for idx, data in enumerate(rawdata):
        time = np.arange(0, data.size) / fs

        scalex, unitx = scale_auto_value(time[xpos_start:xpos_end+1])
        axs[idx, 0].plot(scalex * time, scaley * data, color='k', linewidth=1)
        axs[idx, 0].set_ylabel(f'ADC output [{unity}V]', fontsize=label_fontsize)
        axs[idx, 0].grid()
        axs[idx, 0].tick_params(direction='out', labelsize=tick_textsize)
        axs[idx, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[idx, 0].yaxis.set_minor_locator(AutoMinorLocator())
        axs[idx, 0].plot(scalex * time[spike_xpos[idx]], scaley * data[spike_xpos[idx]], 'r', marker='x', linestyle='None')

    axs[1, 0].set_xlim(xpos_lim)
    axs[1, 0].set_xticks(np.linspace(xpos_lim[0], xpos_lim[1], 5, endpoint=True))
    axs[1, 0].set_xlabel(f'Time [{unitx}s]', fontsize=label_fontsize)
    axs[0, 0].set_title(cell_name[0], loc='left', fontdict={'fontsize': tick_textsize})
    axs[1, 0].set_title(cell_name[1], loc='left', fontdict={'fontsize': tick_textsize})

    # Plot spike shape
    for idx, frame_ch in enumerate(frames):
        xframe_start = np.argwhere(spike_xpos[idx] >= xpos_start).flatten()[0]
        xframe_stop =  np.argwhere(spike_xpos[idx] >= xpos_end).flatten()[0] if spike_xpos[idx][-1] > xpos_end else len(spike_xpos[idx])
        frames_used = frame_ch[xframe_start:xframe_stop+1, :]

        mean_frame = np.median(frames_used, axis=0)
        axs[idx, 1].plot(scaley * np.transpose(frames_used), linewidth=0.5)
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
        path2realsave = os.path.join(get_path_project_start(), path2save)
        save_figure(fig, path2realsave, 'rgc_tdb_transient_example')
    if show_plots:
        plt.show(block=True)


if __name__ == "__main__":
    file_numbers = [19, 36]
    channel_number = [611, 130]

    # --- Loading the src_neuro
    cell_bib = CellSelector(1, 0)

    # --- Start Routine
    print("\nGetting the given data points from the RGC TDB for plotting")
    print(f"... loading the data set")
    rawdata0 = list()
    frames0 = list()
    spike_xpos0 = list()
    cell_name0 = list()
    adc_fs = 0.0
    for idx, file in enumerate(file_numbers):
        ch = channel_number[idx]

        # --- Getting the Data
        data_set = SettingsDATA(
            path="C:\HomeOffice\Data_Neurosignal",
            data_set=7, data_case=0, data_point=file,
            t_range=[0], ch_sel=[],
            fs_resample=40e3
        )
        datahandler = DataLoader(data_set)
        datahandler.do_call()
        datahandler.do_resample()
        data_used = datahandler.get_data()
        del datahandler

        # --- Processing the analogue input (channel specific)
        afe = Pipeline(data_set.fs_resample)
        # spike_xpos = np.floor(data_used.raw_data.evnt_xpos[ch] * afe.fs_adc / afe.fs_ana).astype("int")
        spike_xpos = find_peaks(np.abs(data_used.data_raw[ch]), distance=int(2e-3 * afe.fs_ana),
                                height=0.5* data_used.data_raw[ch].max())[0]
        spike_xpos = np.floor(spike_xpos * afe.fs_adc / afe.fs_ana)

        afe.run_input(data_used.data_raw[ch], spike_xpos)
        adc_fs = afe.fs_adc

        # --- Getting the results
        rawdata0.append(afe.lsb * afe.signals.x_adc)
        frames0.append(afe.lsb * afe.signals.frames_align)
        spike_xpos0.append(afe.signals.x_pos)
        for id in np.unique(data_used.evnt_id[ch]):
            cell_name0.append(cell_bib.get_celltype_name_from_id(int(id)))

    # Delete after runs
    del spike_xpos, idx, ch, file, afe

    print('... plot the results!')
    # Plot transient signals
    plot_rgc_transient_signals(rawdata0, frames0, spike_xpos0, cell_name0, adc_fs, path2save='runs', show_plots=True)
