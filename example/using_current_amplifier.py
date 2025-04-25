import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.analog.amplifier.cur_amp import CurrentAmplifier, DefaultSettingsCUR


if __name__ == "__main__":
    # --- Definiton of signals
    t_end = 50e-3
    fs = 10e3
    i_off = 0.0
    i_pp = [1e-3, 10e-6]
    f_sig = [100.0, 250.0]

    tA = np.linspace(0, t_end, int(t_end * fs), endpoint=True)
    i_in = np.zeros(tA.shape) + i_off
    for idx, f0 in enumerate(f_sig):
        i_in += i_pp[idx] * np.sin(2 * np.pi * f0 * tA)

    # --- Data Processing
    dev_cur = CurrentAmplifier(DefaultSettingsCUR)

    u_out = list()
    u_out.append(i_in)
    u_out.append(dev_cur.transimpedance_amplifier(i_in, 0.0))
    u_out.append(dev_cur.push_amplifier(i_in))
    u_out.append(dev_cur.pull_amplifier(i_in))
    u_out.append(dev_cur.push_pull_amplifier(i_in)[0])
    u_out.append(dev_cur.push_pull_amplifier(i_in)[1])
    u_out.append(dev_cur.push_pull_abs_amplifier(i_in))

    u_dict = ["I_in", "Transimpedance", "Push", "Pull", "Push-Pull (Pos.)", "Push-Pull (Neg.)", "Push-Pull (Abs.)"]

    # --- Plotten
    plt.close('all')
    plt.figure()
    plt_size = [2, 4]
    axs = [plt.subplot(plt_size[0], plt_size[1], 1+idx) for idx in range(plt_size[0] * plt_size[1])]

    for idx, u_sig in enumerate(u_out, start=0):
        axs[idx].plot(tA, u_sig, 'k')
        axs[idx].grid()
        axs[idx].set_title(u_dict[idx])

    axs[6].set_xlabel("Time t / s")
    plt.tight_layout()
    plt.show()