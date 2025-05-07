from denspp.offline.digital.dsp import SettingsFilter, DSP


example_settings0 = SettingsFilter(
    gain=1, fs=2e3,
    n_order=1, f_filt=[1, 200],
    type='iir', f_type='butter', b_type='bandpass',
    t_dly=100e-6,
    q_fac=10
)

example_settings1 = SettingsFilter(
    gain=1, fs=2e3,
    n_order=1, f_filt=[50],
    type='iir', f_type='butter', b_type='notch',
    t_dly=100e-6,
    q_fac=100
)


if __name__ == "__main__":
    dsp = DSP(example_settings1)
    coeffs = dsp.get_filter_coeffs

    coeff_quant = dsp.quantize_coeffs(
        bit_size=12,
        bit_frac=10,
        signed=True
    )
    dsp.coeff_print(12, 10, signed=True)
    dsp.coeff_verilog(12, 10, signed=True)

    dsp.plot_freq_response(
        b=coeffs['b'],
        a=coeffs['a'],
        num_points=2001,
        show_plot=True
    )
