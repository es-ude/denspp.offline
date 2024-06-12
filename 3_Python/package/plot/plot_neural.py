import numpy as np
import matplotlib.pyplot as plt
from package.plot.plot_common import cm_to_inch, save_figure


def plot_signals_neural_cluster(dataset, path2save='') -> None:
    """Plotting the mean waveforms of each cluster"""
    color = 'krbgym'

    num_cl = dataset.frames_me.shape[0]
    num_frame_size = dataset.frames_me.shape[1]-1

    plt.figure()
    axs = [plt.subplot(2, int(np.ceil(num_cl/2)), 1+idx) for idx in range(num_cl)]
    for idx, frame in enumerate(dataset.frames_me):
        axs[idx].plot(frame, color=color[idx])
        axs[idx].grid()
        axs[idx].set_title(dataset.frame_dict[idx])
        axs[idx].set_xlim(0, num_frame_size)
        axs[idx].set_xticks(np.linspace(0, num_frame_size, 5, dtype=int))

    plt.tight_layout(pad=0.5)
    if path2save:
        save_figure(plt, path2save, f"neural_cluster_waveforms")

