import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.analog.amplifier.dly_amp import DelayAmplifier, DefaultSettingsDLY


if __name__ == "__main__":
    settings = DefaultSettingsDLY
    dut = DelayAmplifier(settings)

    # --- Declaration of input
    t_end = 10e-3
    t0 = np.linspace(0, t_end, num=int(t_end * settings.fs_ana), endpoint=True)
    u_off = 0.0
    u_pp = [0.25, 0.3, 0.1]
    f0 = [1e3, 1.8e3, 2.8e3]
    uinp = np.zeros(t0.shape) + u_off
    for idx, peak_val in enumerate(u_pp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * f0[idx])
    uinn = settings.vcm

    uout0 = dut.do_simple_delay(uinp)
    uout1 = dut.do_recursive_delay(uinp)
    uout2 = dut.do_allpass_first_order(uinp)
    uout3 = dut.do_allpass_second_order(uinp, 100.)

    # --- Plotting
    plt.figure()
    plt.plot(t0, uinp, 'r', label="Input")
    plt.plot(t0, uout0, 'k', label="Out (simple)")
    plt.plot(t0, uout1, 'm', label="Out (recursive)")
    plt.plot(t0, uout2, 'y', label="Out (all-pass)")

    plt.xlabel('Time t / s')
    plt.ylabel('Voltage U_x / V')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
