import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from package.plot.plot_common import save_figure, cm_to_inch
from package.data_call.data_call_common import DataController
from package.data_call.data_call_cellbib import CellSelector
from src_data.pipeline_data import Settings, Pipeline


if __name__ == "__main__":
    path2save = '/home/erbsloeh'
    figure_name = 'rgc_example'
    file_numbers = [19, 36]
    channel_number = [611, 130]

    # --- Loading the src_neuro
    cell_bib = CellSelector(1, 0)
    afe_set = Settings()
    gain = afe_set.SettingsAMP.gain
    fs_ana = afe_set.SettingsADC.fs_ana
    fs_adc = afe_set.SettingsADC.fs_adc

    # --- Start Routine
    print("\nGetting the given data points from the RGC TDB for plotting")
    print(f"... loading the datasets")
    rawdata = list()
    frames = list()
    cell_name = list()
    for idx, file in enumerate(file_numbers):
        ch = channel_number[idx]

        # --- Getting the Data
        afe_set.SettingsDATA.data_point = file
        datahandler = DataController(afe_set.SettingsDATA)
        datahandler.do_call()
        datahandler.do_resample()

        spike_xpos = np.floor(datahandler.raw_data.spike_xpos[ch] * fs_adc / fs_ana).astype("int")
        spike_xoff = int(1e-6 * datahandler.raw_data.spike_offset_us[0] * fs_adc)

        # --- Processing the analogue input (channel specific)
        afe = Pipeline(afe_set)
        afe.run_input(datahandler.raw_data.data_raw[ch], spike_xpos, spike_xoff)
        adc_lsb = afe_set.SettingsADC.lsb

        # --- Getting the results
        rawdata.append(adc_lsb * afe.signals.x_adc)
        frames.append(adc_lsb * afe.signals.frames_align)
        for id in np.unique(datahandler.raw_data.cluster_id[ch]):
            cell_name.append(cell_bib.get_celltype_name_from_id(int(id)))

    # Delete after runs
    del spike_xpos, spike_xoff, datahandler, afe, afe_set, idx, ch, file,

    # --- Plotting the results
    fig, axs = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(cm_to_inch(14), cm_to_inch(16)))
    scaley = 1e6 / gain
    tick_textsize = 12
    label_fontsize = 13

    print('... plot the results!')
    # Plot transient signals
    for idx, data in enumerate(rawdata):
        time = np.arange(0, data.size) / fs_adc
        axs[idx, 0].plot(time, scaley * data, color='k', linewidth=1)
        axs[idx, 0].set_ylabel('ADC output [µV]', fontsize=label_fontsize)
        axs[idx, 0].grid()
        axs[idx, 0].tick_params(direction='out', labelsize=tick_textsize)
        axs[idx, 0].xaxis.set_minor_locator(AutoMinorLocator())
        axs[idx, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1, 0].set_xlim([2.3, 3.5])
    axs[1, 0].set_xticks([2.3, 2.6, 2.9, 3.2, 3.5])
    axs[1, 0].set_xlabel('Time [s]', fontsize=label_fontsize)
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
    plt.tight_layout(h_pad=0.05, w_pad=0.05
                     )
    save_figure(fig, path2save, figure_name, ['pdf', 'svg'])
    plt.show(block=True)
