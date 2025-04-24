import numpy as np
from denspp.offline.analog.amplifier.comparator import Comparator, SettingsComparator


ExampleSettingsComparator = SettingsComparator(
    vdd=0.6, vss=-0.6,
    out_analog=False,
    gain=10,
    offset=10e-3,
    noise_dis=1.0e-3,
    hysteresis=0.25
)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    print("TEST")

    cmp = Comparator(ExampleSettingsComparator)
    # --- Defining the input
    n_samples = 10000
    halfspace = np.linspace(ExampleSettingsComparator.vss, ExampleSettingsComparator.vdd, n_samples)
    inp0 = np.concatenate((halfspace, -halfspace), 0)
    inp1 = np.zeros((n_samples, ))

    # --- Defining the output
    out0 = cmp.cmp_normal(inp0, cmp.vcm)
    out1 = cmp.cmp_single_pos_hysteresis(inp0, cmp.vcm)
    out2 = cmp.cmp_single_neg_hysteresis(inp0, cmp.vcm)
    out3 = cmp.cmp_double_hysteresis(inp0, cmp.vcm)

    # --- Plots
    plt.close('all')
    plt.figure()
    axs = [plt.subplot(2, 1, idx+1) for idx in range(2)]

    # --- Transfer function
    axs[0].plot(inp0, out0, label="normal")
    axs[0].plot(inp0, out1, color="r", label="single, pos")
    axs[0].plot(inp0, out2, color="k", label="single, neg")
    axs[0].plot(inp0, out3, color="m", label="double")

    axs[0].grid()
    axs[0].legend()
    if ExampleSettingsComparator.out_analog:
        axs[0].set_ylabel("Output Boolean []")
    else:
        axs[0].set_ylabel("Output voltage [V]")

    # --- Noise Distribution
    axs[1].hist(cmp.get_noise_signal, bins=100, align="left",
                density=True, cumulative=False)
    axs[1].grid()
    axs[1].legend()
    axs[1].set_ylabel("Bins Noise Distribution")
    axs[1].set_xlabel("Input voltage [V]")

    plt.tight_layout()
    plt.show()
