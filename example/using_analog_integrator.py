import matplotlib.pyplot as plt
import numpy as np
from denspp.offline.analog.amplifier.int_ana import IntegratorStage, SettingsINT, RecommendedSettingsINT


if __name__ == "__main__":

    # --- Definition of Inputs
    settings = RecommendedSettingsINT
    f_smp = 10e3
    t_end = 1
    u_off = 0e-3
    upp = [0.1, 0.02]
    f0 = [3, 10]

    # --- Generation of signals
    time = np.linspace(0, t_end, int(t_end * f_smp), endpoint=True)
    u_inp0 = np.zeros(time.shape) + u_off + settings.vcm
    for idx, peak_value in enumerate(upp):
        u_inp0 += peak_value * np.sin(2 * np.pi * time * f0[idx])
    u_inn0 = np.array(settings.vcm)
    i_in0 = u_inp0 - settings.vcm

    # --- DUT (Test condition)
    dev_test = IntegratorStage(settings, f_smp)
    u_out1 = dev_test.do_ideal_integration(u_inp0, u_inn0)
    u_out2 = dev_test.do_opa_volt_integration(u_inp0, u_inn0)
    u_out3 = dev_test.do_cap_curr_integration(i_in0 / 1000)

    # --- Plotting results
    plt.close('all')
    plt.figure()
    plt.plot(time, u_inp0, 'k', label="input")
    plt.plot(time, u_out1, 'b', label="integration (ideal)")
    plt.plot(time, u_out2, 'g', label="integration (real, OPA)")
    plt.plot(time, u_out3, 'r', label="integration (real, cap)")

    plt.grid()
    plt.legend()
    plt.xlabel("Time t / s")
    plt.ylabel("Voltage U_x / V")

    plt.tight_layout()
    plt.show()
