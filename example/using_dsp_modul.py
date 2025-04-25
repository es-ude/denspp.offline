import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.digital.dsp import SettingsFilter, RecommendedSettingsFilter, DSP


if __name__ == "__main__":
    plt.close('all')

    t_end = 200e-3
    settings = RecommendedSettingsFilter
    demo_dsp = DSP(settings)
    num_points = int(settings.fs * t_end)

    f0 = 0.18e3
    f1 = 0.2e3
    df = 0.095e3
    t0 = np.linspace(0, t_end, num_points)
    x0 = np.sin(2 * np.pi * t0 * f0)
    x0[0:int(20e3/f0*2)] = 0
    x0[int(20e3 / f0 * 8):] = 0

    y0 = demo_dsp.time_delay_fir(x0)
    y1 = demo_dsp.time_delay_iir_fir_order(x0, f1, do_plot=True)
    y2 = demo_dsp.time_delay_iir_sec_order(x0, f1, df, do_plot=True)

    plt.figure()
    xscale = 1e3
    plt.plot(xscale * t0, x0, 'k', marker='.', label='x[n]')
    plt.plot(xscale * t0, y0, 'r', label='y[n] (FIR)')
    plt.plot(xscale * t0, y1, 'b', label='y[n] (IIR 1st order)')
    plt.plot(xscale * t0, y2, 'g', label='y[n] (IIR 2nd order)')

    plt.xlabel('Time / ms')
    plt.ylabel('Y(t)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
