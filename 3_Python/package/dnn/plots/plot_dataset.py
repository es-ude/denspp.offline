import numpy as np
from matplotlib import pyplot as plt
from package.plot_helper import cm_to_inch
from package.metric.data import calculate_snr


def plot_frames_dataset(data: dict, take_samples=500, do_norm: bool=False,
                        plot_norm: bool=False, plot_show: bool=False, add_subtitle: bool=False) -> None:
    """Plotting the frames for different classes into one figure
    Args:
        data:           Dictionary with spike frames, peak amplitudes and labels
        take_samples:   Only take random N samples from each class
        do_norm:        Do normalization (minmax)
        plot_norm:      Plot option for do normalization on input data
        plot_show:      Plot option for blocking and showing the plots
        add_subtitle:   Adding a subtitle with further informations
    Return:
        None
    """
    frame_raw = data['data']
    frame_true = data['label']
    frame_dict = data['dict']
    frame_peak = data['peak'] / np.abs(frame_raw.min()) if 'peak' in data.keys() else np.max(np.abs(data['data']), 1)

    # --- Figure #1: Spike Frames
    num_rows = 2
    num_cols = int(np.ceil(len(frame_dict)/num_rows))

    cluster_id, cluster_cnt = np.unique(frame_true, return_counts=True)
    _, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(cm_to_inch(14), cm_to_inch(16)))
    for id, key in enumerate(frame_dict):
        pos_id = np.argwhere(frame_true == id).flatten()
        pos_random = pos_id[np.random.randint(0, pos_id.size, take_samples)]
        scale = 1 if not do_norm else np.repeat(np.expand_dims(frame_peak[pos_random], -1), frame_raw.shape[-1], -1)

        frame_raw0 = frame_raw[pos_random, :] / scale
        frame_mean = np.mean(frame_raw0, axis=0)
        snr_class = list()
        for frame in frame_raw0:
            snr_class.append(calculate_snr(frame, frame_mean))

        axs[int(id/num_cols), id % num_cols].plot(np.transpose(frame_raw0), linewidth=0.5)
        axs[int(id/num_cols), id % num_cols].plot(frame_mean, 'k', linewidth=2.0)
        axs[int(id/num_cols), id % num_cols].grid()
        if not add_subtitle:
            axs[int(id/num_cols), id % num_cols].set_title(key, fontsize=13)
        else:
            text = f'{key}\nnum={cluster_cnt[id]}, median(SNR) = {np.mean(np.array(snr_class)):.2f} dB'
            axs[int(id / num_cols), id % num_cols].set_title(text, fontsize=13)

    axs[0, 0].set_xlim([0, frame_raw.shape[-1]-1])
    axs[0, 0].set_xticks(np.linspace(0, frame_raw.shape[-1]-1, 7, endpoint=True, dtype=np.uint16))
    if plot_norm:
        axs[0, 0].set_ylabel("Spike Norm. Value", fontsize=13)
        axs[1, 0].set_ylabel("Spike Norm. Value", fontsize=13)
    else:
        axs[0, 0].set_ylabel("Spike Voltage [µV]", fontsize=13)
        axs[1, 0].set_ylabel("Spike Voltage [µV]", fontsize=13)

    for idx in range(num_cols):
        axs[1, idx].set_xlabel("Spike Frame Position", fontsize=13)
    plt.tight_layout()

    # --- Figure #2: Histogram - Spike Frame Peak Amplitude
    _, axs = plt.subplots(1, 2, sharex=True)
    axs[0].hist(frame_peak, bins=101)
    axs[1].hist(frame_peak, bins=101, density=True, cumulative=True)

    axs[0].set_xlim([0, frame_peak.max()])
    axs[0].set_ylabel('Bins', fontsize=13)
    axs[0].set_xlabel('Spike Peak Amplitude [µV]', fontsize=13)
    axs[1].set_ylabel('Density', fontsize=13)
    axs[1].set_xlabel('Spike Peak Amplitude [µV]', fontsize=13)

    for ax in axs:
        ax.grid()
    plt.tight_layout()

    if plot_show:
        plt.show(block=True)
