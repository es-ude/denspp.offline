import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.analog.amplifier.comparator import Comparator, SettingsCOMP


ExampleSettingsComparator = SettingsCOMP(
    vdd=0.6, vss=-0.6,
    gain=100,
    offset=5e-3,
    noise_dis=10e-3,
    hysteresis=0.25,
    out_analog=False,
    out_invert=False
)


if __name__ == "__main__":
    # --- Defining the input
    n_runs = 201
    n_samples = 100
    scale = 0.01
    halfspace = np.linspace(ExampleSettingsComparator.vss, ExampleSettingsComparator.vdd, n_runs)
    voltage_inp = scale* np.concatenate((halfspace, -halfspace), 0)
    del halfspace

    # --- Start Experiment
    prob_false = np.zeros_like(voltage_inp)
    prob_true = np.zeros_like(voltage_inp)

    cmp = Comparator(ExampleSettingsComparator)
    for idx, vinp in enumerate(voltage_inp):
        out0 = np.array([cmp.cmp_normal(vinp, cmp.vcm) for n in range(n_samples)]).flatten()
        prob_false[idx] = np.sum(out0 == False) / n_samples
        prob_true[idx] = np.sum(out0 == True) / n_samples

    # --- Plots
    plt.close('all')
    plt.figure()
    plt.plot(voltage_inp, 100* prob_false, color='k', label='false')
    plt.plot(voltage_inp, 100* prob_true, color='r', label='true')
    plt.ylabel('Probability for Event [%]')
    plt.xlabel('Input Voltage [V]')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show(block=True)
