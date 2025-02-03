import matplotlib.pyplot as plt
from denspp.offline.data_generator.waveform_generator import WaveformGenerator


if __name__ == "__main__":
    num_elements = 13
    time_points = [0.2 + 0.4 * idx for idx in range(num_elements)]
    time_duration = [0.1 for idx in range(num_elements)]
    time_wfg = [idx for idx in range(num_elements)]
    polarity_cathodic = [False for idx in range(num_elements)]
    # polarity_cathodic = [idx % 3 == 0 for idx in range(num_elements)]

    wfg_generator = WaveformGenerator(50e3)
    signal0 = wfg_generator.generate_waveform(time_points, time_duration, time_wfg, polarity_cathodic)
    signal1 = wfg_generator.generate_biphasic_waveform(0, 0.1, 0, 0.2, 0.05, True, True)
    wfg_generator.check_charge_balancing(signal1['sig'])

    # --- Plotting: All waveforms
    plt.figure()
    plt.plot(signal0['time'], signal0['sig'], 'k')
    plt.xlabel("Time t / s")
    plt.ylabel("Signal y(t)")
    plt.grid()
    plt.tight_layout()

    # --- Plotting: Biphasic waveform
    plt.figure()
    plt.plot(signal1['time'], signal1['sig'], 'k')
    plt.xlabel("Time t / s")
    plt.ylabel("Signal y(t)")
    plt.grid()
    plt.tight_layout()
    plt.show()
