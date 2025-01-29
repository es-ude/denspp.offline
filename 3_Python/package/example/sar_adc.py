import numpy as np
import matplotlib.pyplot as plt

from package.plot_helper import scale_auto_value
from package.analog.adc.adc_sar import SARADC
from package.analog.adc import SettingsADC
from package.analog.dev_noise import noise_real
from package.data_process.transformation import do_fft


def example_sar_adc() -> None:
    """"""
    set_adc = SettingsADC(
        vdd=0.6, vss=-0.6,
        fs_ana=200e3, fs_dig=20e3, osr=10,
        dvref=0.1, Nadc=12,
        type_out="signed"
    )

    adc0 = SARADC(set_adc)

    t_end = 1
    tA = np.arange(0, t_end, 1/set_adc.fs_ana)
    tD = np.arange(0, t_end, 1/set_adc.fs_dig)

    # --- Input signal
    upp = 0.8 * set_adc.dvref
    fsine = 100
    uin = upp * np.sin(2 * np.pi * tA * fsine)
    uin += noise_real(tA.size, tA.size, -120, 1, 0.6)[0]

    # --- ADC output
    uadc_hs = adc0.adc_sar_ns_order_one(uin)[0]
    uadc = adc0._do_downsample(uadc_hs)
    freq, Yadc = do_fft(uadc_hs, set_adc.fs_adc)

    # --- Plotting
    plot_sar_results(tA, uin, tD, uadc, freq, Yadc)


def plot_sar_results(tA: np.ndarray, uin: np.ndarray, tD: np.ndarray, uadc: np.ndarray,
                     freq: np.ndarray, Yadc: np.ndarray) -> None:
    """"""
    plt.close('all')
    plt.figure()
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313)

    vscale, vunit = scale_auto_value(uin)
    ax1.plot(tA, vscale * uin)
    ax1.set_ylabel(f'U_in [{vunit}V]')
    ax1.set_xlim([100e-3, 150e-3])

    ax2.plot(tD, uadc)
    ax2.set_ylabel('X_adc []')
    ax2.set_xlabel('Time t [s]')

    ax3.semilogx(freq, 20 * np.log10(Yadc))
    ax3.set_ylabel('X_adc []')
    ax3.set_xlabel('Frequency f [Hz]')

    plt.tight_layout()
    plt.show(block=True)
