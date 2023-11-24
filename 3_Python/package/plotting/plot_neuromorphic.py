import numpy as np
import matplotlib.pyplot as plt
from package.plotting.plot_common import cm_to_inch, save_figure


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

    plt.show(block=True)
