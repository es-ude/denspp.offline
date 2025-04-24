import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.analog.amplifier.comparator import Comparator, SettingsComparator


ExampleSettingsComparator = SettingsComparator(
    vdd=0.6, vss=-0.6,
    gain=100,
    offset=-100e-3,
    noise_dis=1.0e0,
    hysteresis=0.25,
    out_analog=False,
    out_invert=False
)


if __name__ == "__main__":
    # --- Defining the input
    n_samples = 10000
    halfspace = np.linspace(ExampleSettingsComparator.vss, ExampleSettingsComparator.vdd, n_samples)
    voltage_inp = np.concatenate((halfspace, -halfspace), 0)
    del halfspace

    # --- Defining the output
    cmp = Comparator(ExampleSettingsComparator)
    out0 = cmp.cmp_ideal(voltage_inp, cmp.vcm)
    out1 = cmp.cmp_normal(voltage_inp, cmp.vcm)
    out2 = cmp.cmp_single_pos_hysteresis(voltage_inp, cmp.vcm)
    out3 = cmp.cmp_single_neg_hysteresis(voltage_inp, cmp.vcm)
    out4 = cmp.cmp_double_hysteresis(voltage_inp, cmp.vcm)

    # --- Plots
    plt.close('all')
    plt.figure()
    axs = [plt.subplot(3, 2, idx+1) for idx in range(6)]

    # --- Transfer function
    axs[0].plot(voltage_inp, out0, color='k', label="ideal")
    axs[1].plot(voltage_inp, out1, color='k', label="normal")
    axs[2].plot(voltage_inp, out2, color='k', label="single, pos")
    axs[3].plot(voltage_inp, out3, color='k', label="single, neg")
    axs[4].plot(voltage_inp, out4, color='k', label="double")
    axs[4].set_xlabel("Input voltage [V]")

    for ax in axs[0:-1]:
        ax.grid()
        ax.legend()
        if not ExampleSettingsComparator.out_analog:
            ax.set_ylabel("Output []")
        else:
            ax.set_ylabel("Output [V]")

    # --- Noise Distribution
    axs[-1].hist(cmp.get_noise_signal, color='k', bins=100,
                 align="left", density=True, cumulative=False)
    axs[-1].grid()
    axs[-1].set_ylabel("Bins Noise Distribution")
    axs[-1].set_xlabel("Input voltage [V]")

    plt.tight_layout()
    plt.show()
