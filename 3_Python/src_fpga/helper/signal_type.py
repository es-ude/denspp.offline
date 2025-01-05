import numpy as np
from scipy.signal import sawtooth
from package.data_process.transformation import do_fft
from package.metric.data import (calculate_error_mape, calculate_error_mae, calculate_error_mbe)
from package.metric.electrical import calculate_total_harmonics_distortion
from package import calculate_cosine_similarity


def checking_binary_limits_violation(data: np.ndarray, bitwidth: int, do_signed: bool) -> np.ndarray:
    """Function for checking if data has some binary limit violations and correct them
    Args:
        data:       Numpy array with data for checking
        bitwidth:   Used bitwidth for checking
        do_signed:  Output is signed (True) or unsigned (False)
    Return:
        Numpy array with corrected binary data
    """
    chck_lim = [-(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1) - 1)] if do_signed else [0, ((2 ** bitwidth) - 1)]
    if data.max() > chck_lim[1]:
        xpos = np.argmax(data)
        data[xpos] = chck_lim[1]
    if data.min() < chck_lim[0]:
        xpos = np.argmin(data)
        data[xpos] = chck_lim[0]
    return data


def generation_sinusoidal_waveform(bitsize_lut: int, f_rpt: float, f_sine: float,
                                   bitsize_chck=-1, do_signed=False, do_optimized=False) -> np.ndarray:
    """Generating the sinusoidal waveform for building LUT file
    Args:
        bitsize_lut:    Used quantization level for generating sinusoidal waveform LUT
        f_rpt:          Frequency of the timer interrupt
        f_sine:         Target frequency of the sinusoidal waveform at output
        bitsize_chck:   Used bitwidth for checking [Default: -1 --> take bitsize_lut]
        do_signed:      Output is signed or unsigned
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
    Return:
        Numpy array with waveform [np.int32]
    """
    reduced_samples = 1.0 if not do_optimized else 0.25
    num_lutsine = int(reduced_samples * f_rpt / f_sine) + (1 if do_optimized else 0)

    # Generating sine waveform as template
    x0 = np.linspace(0, reduced_samples * 2 * np.pi, num_lutsine)
    offset = 0 if do_signed else 1
    sine_lut = (2 ** (bitsize_lut - 1) * (offset + np.sin(x0)))

    # Limitations to output range
    bitsize_lut = bitsize_lut if bitsize_chck == -1 else bitsize_chck
    sine_lut_chck = checking_binary_limits_violation(sine_lut, bitsize_lut, do_signed)
    return np.array(sine_lut_chck, dtype=np.int32)


def generate_triangular_waveform(f_sig: float, fs: float, no_periods=1, bit=16) -> [np.ndarray, np.ndarray]:
    """Function to generate a triangular waveform for testing digital filters

    Args:
        f_sig:          Frequency of the triangular input
        fs:             Desired sampling frequency
        no_periods:     Number of periods
    """
    t_end = no_periods / f_sig
    time0 = np.arange(0, t_end, 1 / fs)
    offset = (2 ** (bit-1))-1

    data_in = np.array(offset * (1 + 0.9 * sawtooth(2 * np.pi * time0 * f_sig, 0.5)), dtype='uint16')

    return time0, data_in


def generate_rectangular_waveform(f_sig: float, fs: float, no_periods=1, bit=16) -> [np.ndarray, np.ndarray]:
    """Function to generate a rectangular waveform for testing digital filters
    Args:
        f_sig:          Frequency of the triangular input
        fs:             Desired sampling frequency
        no_periods:     Number of periods
    """
    t_end = no_periods / f_sig
    time0 = np.arange(0, t_end, 1 / fs)
    offset = (2 ** (bit-1))-1

    ref_sig = np.sin.step(2 * np.pi * time0 * f_sig, 0.5)
    ref_sig[ref_sig > 0] = 1.0
    ref_sig[ref_sig <= 0] = -1.0
    data_in = np.array(offset * (1 + 0.9 * ref_sig), dtype='uint16')
    return time0, data_in


def creating_timeseries_waveform(f_smp: float, f_upd: float, t_end: float, waveform_lut: np.ndarray, lut_optimized: bool) -> dict:
    """Creating a transient signal from waveform LUT
    Args:
        f_smp:          Sampling rate of transient signals
        f_upd:          Update rate for calling LUT table
        t_end:          End of transient time analysis
        waveform_lut:   Numpy array with waveform LUT
        lut_optimized:  Option if LUT is optimized (True, mirrored signal call) or not (False)
    Return:
        Dictionary with time and signal numpy array
    """
    signal_out = dict()
    signal_out.update({'fs': f_smp, 't': np.linspace(0, t_end, int(t_end * f_smp), endpoint=True)})

    # --- Calculating transient signal
    if lut_optimized:
        lut_used = np.concatenate((waveform_lut, np.flip(waveform_lut[:-1], axis=0),
                                   -waveform_lut[1:], -np.flip(waveform_lut[1:-1], axis=0)))
    else:
        lut_used = waveform_lut[:-1]
    signal = np.zeros(signal_out['t'].shape)
    for pos, idx in enumerate(np.floor(signal_out['t'] * f_upd)):
        pos_lut = int(idx % lut_used.size)
        signal[pos] = lut_used[pos_lut]
    signal += np.random.randint(-1, 1, size=signal.size)
    signal_out.update({'wvf': signal})

    return signal_out


def calculate_metrics(signal: dict, signal_reference: np.ndarray, fsine: float, N_harmonics=6) -> dict:
    """Calculating metrics
    Args:
        signal:             Dictionary with signal from transient and spectral analys
        signal_reference:   Transient signal reference for calculating metrics
        fsine:              Target frequency of the sinusoidal waveform at output
        N_harmonics:        Number of harmonics for calculation
    Return:
        Dictionary with different metrics
    """
    metrics = dict()
    metrics.update({'MBE': calculate_error_mbe(signal['wvf'], signal_reference, False),
                    'MAE': calculate_error_mae(signal['wvf'], signal_reference, False),
                    'MAPE': calculate_error_mape(signal['wvf'], signal_reference, False)})

    # --- Similarity Test
    xpos0 = np.argwhere(signal['t'] >= 2/fsine).flatten()[0]
    xpos1 = np.argwhere(signal['t'] >= 12/fsine).flatten()[0]
    metrics.update({'SIM': calculate_cosine_similarity(signal['wvf'][xpos0:xpos1], signal_reference[xpos0:xpos1], False)})

    # --- Total Harmonics Distortion
    thd = calculate_total_harmonics_distortion(signal['freq'], signal['Y'], fsine, N_harmonics)
    metrics.update({'THD': thd[0], 'THD_POS': thd[1]})

    return metrics


def plot_waveform_results(signal: dict, lut_waveform: np.ndarray, tran_ref: np.ndarray,
                          block_plot=False, metrics=None) -> None:
    """Plotting the transient and spectral results
    Args:
        signal:
        lut_waveform:
        tran_ref:       Transient signal reference for calculating metrics
        block_plot:     Plotting and blocking the plots
        metrics:        Dictionary with metrics
    Return:
        Numpy array with reference signal
    """
    fsine_used = 1e3
    dxf = 8
    fig, axis = plt.subplots(3, 1, gridspec_kw=dict(hspace=0.35, wspace=0.05))

    # --- LUT Waveform
    axis[0].plot(lut_waveform, 'k', marker='.')
    axis[0].set_xlabel('LUT position')
    axis[0].set_ylabel('LUT value')
    axis[0].set_xlim([0, lut_waveform.size-1])
    # --- Transient signals
    axis[1].plot(signal['t'], signal['wvf'] / signal['wvf'].max(), 'k', marker='.', label='input')
    axis[1].plot(signal['t'], tran_ref / tran_ref.max(), 'r', marker='.', label='reference')
    axis[1].legend()
    axis[1].set_xlabel('Time t / s')
    axis[1].set_ylabel('Norm. signal / []')
    axis[1].set_xlim([1/fsine_used, 5/fsine_used])
    # --- Spectral results
    axis[2].semilogy(signal['freq'], signal['Y'] / signal['Y'].max(), 'k', label='raw')
    if isinstance(metrics, dict):
        if 'THD_POS' in metrics.keys():
            pos_f = metrics['THD_POS']
            for idx, freq0 in enumerate(pos_f):
                xpos = np.argwhere(signal['freq'] >= freq0).flatten()[0]
                axis[2].semilogy(signal['freq'][xpos-dxf:xpos+dxf], signal['Y'][xpos-dxf:xpos+dxf] / signal['Y'].max(),
                                 ('r' if idx == 0 else 'b'), label=('Base' if idx == 0 else 'Harmonic'))
            axis[2].legend()

            thd = metrics['THD']
            mae = metrics['MBE']
            mape = metrics['MAPE']
            sim = metrics['SIM']
            axis[2].set_title(f'MBE = {mae:.4f}, MAPE = {mape:.4f}, Similarity = {100 * sim:.3f} %, THD = {thd:.2f} dB')


    axis[2].set_xlabel('Frequency f / Hz')
    axis[2].set_ylabel('Norm. peak value')
    axis[2].set_xlim([0.5 * fsine_used, 10 * fsine_used])

    for ax in axis:
        ax.grid()

    plt.tight_layout(pad=0.1)
    if block_plot:
        plt.show(block=True)


def plot_metric_result(result: dict, metric: str, dependent_variable: list, name_dependent_var: str) -> None:
    metric_list = []
    for i in result:
        metric_list.append(result[i][metric])
    plt.plot(dependent_variable, metric_list)
    plt.ylabel(metric)
    plt.xlabel(name_dependent_var)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # --- Settings
    bitwidth = [16]
    fs = 2e6
    Ntmp = [9, 11, 13, 15, 17, 21, 23, 25, 27, 31]
    Ntmp = np.arange(5, 90)
    fsine = 1e3
    t_end = 0.5
    lut_optimized = False
    do_plot = False

    results = dict()
    print(f"\nLUT generation and analysis for different configurations\n==============================================")
    for bitwidth_used in bitwidth:
        for Nsine in Ntmp:
            # --- Transient signal generation incl. reference
            lut_waveform_opt = generation_sinusoidal_waveform(bitwidth_used, Nsine* fsine, fsine, do_signed=True, do_optimized=True)
            lut_waveform_full = generation_sinusoidal_waveform(bitwidth_used, Nsine * fsine, fsine, do_signed=True, do_optimized=False)
            lut_used = lut_waveform_opt if lut_optimized else lut_waveform_full

            tran_signals = creating_timeseries_waveform(fs, (Nsine-1)* fsine, t_end, lut_used, lut_optimized=lut_optimized)
            tran_ref = (2 ** (bitwidth_used - 1) - 1) * np.sin(2 * np.pi * fsine * tran_signals['t'] - np.pi / Nsine)

            # --- Spectral Transformation and metric calculation
            freq, Y = do_fft(tran_signals['wvf'], tran_signals['fs'])
            tran_signals.update({'freq': freq, 'Y': Y})

            # --- Plotting
            metrics = calculate_metrics(tran_signals, tran_ref, fsine)
            if do_plot:
                plot_waveform_results(tran_signals, lut_used, tran_ref,
                                      Nsine == Ntmp[-1] and bitwidth_used == bitwidth[-1], metrics)

            # --- Print results
            results.update({f'run_b{bitwidth_used:02d}_l{Nsine:03d}': metrics})
            sim = metrics['SIM']
            thd = metrics['THD']
            mae = metrics['MBE']
            mape = metrics['MAPE']
            print(f"LUT_SIZE = {Nsine} @ {bitwidth_used} bit: MAE = {mae:.4f}, MAPE = {mape:.4f}, "
                  f"Similarity = {100 * sim:.4f} % and THD = {thd:.2f} dB")

    plot_metric_result(results, "THD", Ntmp, "Number of steps in LUT")