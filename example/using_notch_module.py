import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.plot_helper import get_plot_color, scale_auto_value
from denspp.offline.data_process.transformation import do_fft
from denspp.offline.analog.dev_handler import generate_test_signal
from denspp.offline.digital.dsp import SettingsFilter, DSP


example_settings0 = SettingsFilter(
    gain=1, fs=10e3,
    n_order=1, f_filt=[1, 200],
    type='iir', f_type='butter', b_type='bandpass',
    t_dly=0.0
)

example_settings1 = SettingsFilter(
    gain=1, fs=10e3,
    n_order=10, f_filt=[500],
    type='iir', f_type='butter', b_type='notch',
    t_dly=0.0
)


if __name__ == "__main__":
    dsp = DSP(example_settings1)
    coeffs = dsp.get_filter_coeffs

    # Quantized output of coeffs
    coeff_quant = dsp.quantize_coeffs(
        bit_size=12,
        bit_frac=10,
        signed=True
    )
    dsp.coeff_print(12, 10, signed=True)
    dsp.coeff_verilog(12, 10, signed=True)

    # Plot frequency response
    dsp.plot_freq_response(
        b=coeffs['b'],
        a=coeffs['a'],
        num_points=2001,
        show_plot=True
    )

    # Really plot
    time, signal = generate_test_signal(
        t_end=0.1,
        fs=dsp.settings.fs,
        upp=[0.5, 1.0, 0.5],
        fsig=[250, 500, 1000],
        uoff=0.0
    )
    ufilt = dsp.filter(signal)

    plt.figure()
    plt.plot(time, signal, color=get_plot_color(0), label='raw')
    plt.plot(time, ufilt, color=get_plot_color(1), label='filtered')
    plt.grid()
    plt.legend()

    freq, Y = do_fft(
        y=ufilt,
        fs=dsp.settings.fs,
        method_window='hanning'
    )
    plt.figure()
    plt.plot(freq, Y, color=get_plot_color(0), label='raw')
    plt.grid()
    plt.legend()
    plt.show(block=True)
