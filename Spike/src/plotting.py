import matplotlib.pyplot as plt
import numpy as np
from main import sorting_signals

def resultsAFE(fs_ana: int, fs_adc: int, signals: sorting_signals):
    uin = signals.u_spk[0]
    xadc = signals.x_adc
    xsda = signals.x_sda
    xthr = signals.x_thr
    ticks = signals.spike_ticks

    tA = np.arange(0, uin.size, 1) / fs_ana
    tD = np.arange(0, xadc.size, 1) / fs_adc

    # --- Plotting
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(tA, 1e6* uin, 'k')
    plt.ylabel("$U_{in}$ (µV)")

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(tD, xadc, 'k')
    plt.ylabel("$X_{adc}$")

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(tD, xsda, 'k')
    ax3.plot(tD, xthr, 'r')
    plt.ylabel("$X_{sda}$")


    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    idx = 0
    for wave in ticks:
        ax4.plot(tD, 1.25*idx + wave)
        idx += 1

    plt.ylabel("Spike Ticks")
    plt.xlabel("Time t (s)")
    plt.show()

def resultsFEC(signals: sorting_signals):
    framesIn = np.rot90(signals.frames_orig, k=3)
    framesOut = np.rot90(signals.frames_align, k=3)

    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(framesIn)

    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(framesOut)

    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(signals.features[0,:], signals.features[1,:])

    plt.show()
