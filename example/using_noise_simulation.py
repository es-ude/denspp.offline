import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.analog.dev_noise import ProcessNoise, SettingsNoise
from denspp.offline.preprocessing.transformation import do_fft


if __name__ == "__main__":
    fs0 = 2e3
    settings_noise = SettingsNoise(
        temp=300,
        wgn_dB=-70,
        Fc=10,
        slope=0.6
    )
    handler_noise = ProcessNoise(
        settings=settings_noise,
        fs_ana=fs0
    )

    # --- Signal generation
    fs0 = 2e3
    t = np.arange(0, 2e7, 1) / fs0
    noise_w = handler_noise.gen_noise_awgn_pwr(t.size)
    noise_f = handler_noise.gen_noise_flicker_volt(t.size)
    noise_r = handler_noise.gen_noise_real_pwr(t.size)

    # --- Plotting
    scale = [1e3, 1e3, 1e3]
    plt.close('all')
    plt.figure()
    axs = [plt.subplot(3, 3, idx+1) for idx in range(9)]

    axs[0].set_title('Flicker noise')
    axs[0].plot(t, scale[0] * noise_f)
    axs[3].hist(scale[0] * noise_f, bins=100, density=True)
    freq0, psd_real0 = do_fft(noise_f, fs0)
    axs[6].loglog(freq0, psd_real0)

    axs[1].set_title('White noise')
    axs[1].plot(t, scale[1] * noise_w)
    axs[4].hist(scale[1] * noise_w, bins=100, density=True)
    freq1, psd_real1 = do_fft(noise_w, fs0)
    axs[7].loglog(freq1, psd_real1)

    axs[2].set_title('Real noise')
    axs[2].plot(t, scale[2] * noise_r)
    axs[5].hist(scale[2] * noise_r, bins=100, density=True)
    freq2, psd_real2 = do_fft(noise_r, fs0)
    axs[8].loglog(freq2, psd_real2)

    # --- Labeling
    for ax in axs:
        ax.grid()

    for idx in range(0, 3):
        axs[idx].set_xlabel('Time t / s')
        axs[idx].set_ylabel('Voltage U / mV')

    for idx in range(6, 9):
        axs[idx].set_xlabel('Frequency f / Hz')
        axs[idx].set_ylabel('e_n / V/sqrt(Hz)')

    plt.tight_layout()
    plt.show()
