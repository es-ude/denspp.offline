import numpy as np
from matplotlib import pyplot as plt

from denspp.offline.metric import calculate_snr
from denspp.offline.dnn import DatasetFromFile
from denspp.offline.plot_helper import (
    save_figure,
    scale_auto_value,
    cm_to_inch,
    get_textsize_paper
)


def plot_frames_dataset(
        data: DatasetFromFile, take_samples: int=500, do_norm: bool=False,
        add_subtitle: bool=False, path2save: str='', show_plot: bool=False,
) -> None:
    """Plotting the frames for different classes into one figure
    Args:
        data:           Dictionary with spike frames, peak amplitudes and labels
        take_samples:   Only take random N samples from each class
        do_norm:        Do normalization (minmax)
        add_subtitle:   Adding a subtitle with further information
        path2save:      Path to save figure
        show_plot:      Plot option for blocking and showing the plots
    Return:
        None
    """
    frame_raw = data.data
    frame_true = data.label
    frame_dict = data.dict
    frame_peak = np.max(np.abs(data.data), axis=1)
    scaley, unity = scale_auto_value(data.data)

    # --- Figure #1: Spike Frames
    num_rows = 2
    num_cols = int(np.ceil(len(frame_dict)/num_rows))

    cluster_id, cluster_cnt = np.unique(frame_true, return_counts=True)
    fig, axs = plt.subplots(num_rows, num_cols, sharex=True, sharey=True, figsize=(cm_to_inch(14), cm_to_inch(16)))
    for id, key in enumerate(frame_dict):
        pos_id = np.argwhere(frame_true == id).flatten()
        pos_random = pos_id[np.random.randint(0, pos_id.size, take_samples)]
        scale = 1 if not do_norm else np.repeat(np.expand_dims(frame_peak[pos_random], -1), frame_raw.shape[-1], -1)

        frame_raw0 = frame_raw[pos_random, :] / scale
        frame_mean = np.mean(frame_raw0, axis=0)
        snr_class = list()
        for frame in frame_raw0:
            snr_class.append(calculate_snr(frame, frame_mean))

        axs[int(id/num_cols), id % num_cols].plot(scaley * np.transpose(frame_raw0), linewidth=0.5)
        axs[int(id/num_cols), id % num_cols].plot(scaley * frame_mean, 'k', linewidth=2.0)
        axs[int(id/num_cols), id % num_cols].grid()
        if not add_subtitle:
            axs[int(id/num_cols), id % num_cols].set_title(key, fontsize=get_textsize_paper())
        else:
            text = f'{key}\nnum={cluster_cnt[id]}, median(SNR) = {np.mean(np.array(snr_class)):.2f} dB'
            axs[int(id / num_cols), id % num_cols].set_title(text, fontsize=get_textsize_paper())

    axs[0, 0].set_xlim([0, frame_raw.shape[-1]-1])
    axs[0, 0].set_xticks(np.linspace(0, frame_raw.shape[-1]-1, 7, endpoint=True, dtype=np.uint16))
    if do_norm:
        axs[0, 0].set_ylabel("Spike Norm. Value", fontsize=get_textsize_paper())
        axs[1, 0].set_ylabel("Spike Norm. Value", fontsize=get_textsize_paper())
    else:
        axs[0, 0].set_ylabel(f"Spike Voltage [{unity}V]", fontsize=get_textsize_paper())
        axs[1, 0].set_ylabel(f"Spike Voltage [{unity}V]", fontsize=get_textsize_paper())

    for idx in range(num_cols):
        axs[1, idx].set_xlabel("Spike Frame Position", fontsize=get_textsize_paper())
    plt.tight_layout()

    # --- Figure #2: Histogram - Spike Frame Peak Amplitude
    _, axs = plt.subplots(1, 2, sharex=True)
    axs[0].hist(frame_peak, bins=101)
    axs[1].hist(frame_peak, bins=101, density=True, cumulative=True)

    axs[0].set_xlim([0, frame_peak.max()])
    axs[0].set_ylabel('Bins', fontsize=get_textsize_paper())
    axs[0].set_xlabel(f'Spike Peak Amplitude [{unity}V]', fontsize=get_textsize_paper())
    axs[1].set_ylabel('Density', fontsize=get_textsize_paper())
    axs[1].set_xlabel(f'Spike Peak Amplitude [{unity}V]', fontsize=get_textsize_paper())

    for ax in axs:
        ax.grid()
    plt.tight_layout()
    if path2save:
        save_figure(fig, path=path2save, name='dataset_frames')
    if show_plot:
        plt.show(block=True)


def plot_mnist_dataset(
        data: np.ndarray, label: np.ndarray,
        title:str="", path2save:str="", show_plot:bool=False
) -> None:
    """Plotting examples of all labels from MNIST dataset
     :param data:               Numpy array with data content
     :param label:              Numpy array with label content
     :param title:              String with title appendix
     :param path2save:          Path to save figure
     :param show_plot:          Boolean for showing plot
     :return:                   None
     """
    plt.figure()
    axs = [plt.subplot(3, 3, idx + 1) for idx in range(9)]
    print(label, type(label), type(label[0]))
    for idx, ax in enumerate(axs):
        pos_num = int(np.argwhere(label == idx).flatten()[0])
        ax.imshow(data[pos_num], cmap=plt.get_cmap('gray'))

        ax.set_title(f"Label = {idx}")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if path2save:
        save_figure(plt, path=path2save, name=f"dataset_mnist{title}")
    if show_plot:
        plt.show(block=True)


def plot_waveforms_dataset(
        dataset: DatasetFromFile, num_samples_class: int=1,
        path2save: str='', show_plot: bool=True
) -> None:
    """Plotting examples of all labels from waveform dataset
     :param dataset:            Dataclass with loaded dataset using DatasetLoader
     :param num_samples_class:  Integer with number of samples to plot from dataset (per class)
     :param path2save:          Path to save figure
     :param show_plot:          Boolean for showing plot
     :return:                   None
     """
    class_id = np.unique(dataset.label)

    fig, axs = plt.subplots(3, int(np.ceil(class_id.size / 3)), sharex=True)
    use_subplot = [False for _ in range(axs.shape[0] * axs.shape[1])]
    for idx in class_id:
        use_subplot[idx] = True
        num_column = int(np.floor(idx / axs.shape[1]))
        val_range = np.argwhere(dataset.label == idx).flatten()
        sample_idx = np.unique(np.random.randint(low=val_range[0], high=val_range[-1], size=num_samples_class))
        axs[num_column, idx % axs.shape[1]].plot(dataset.data[sample_idx, :].T, label=dataset.dict[idx], color='gray')
        axs[num_column, idx % axs.shape[1]].plot(np.mean(dataset.data[val_range, :], axis=0), label=dataset.dict[idx], color='k')
        axs[num_column, idx % axs.shape[1]].set_title(f'{dataset.dict[idx]}')
        axs[num_column, idx % axs.shape[1]].axis('off')
    for idx in [i for i, val in enumerate(use_subplot) if not val]:
        num_column = int(np.floor(idx / axs.shape[1]))
        axs[num_column, idx % axs.shape[1]].axis('off')

    plt.tight_layout()
    if path2save:
        save_figure(fig, path=path2save, name='dataset_waveforms')
    if show_plot:
        plt.show(block=True)
