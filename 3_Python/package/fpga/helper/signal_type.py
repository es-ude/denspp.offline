import numpy as np


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
    chck_val = bitsize_lut if bitsize_chck == -1 else bitsize_chck
    chck_lim = [-(2 ** (chck_val - 1)), (2 ** (chck_val - 1) - 1)]
    if sine_lut.max() > chck_lim[1]:
        xpos = np.argmax(sine_lut)
        sine_lut[xpos] = chck_lim[1]
    if sine_lut.min() < chck_lim[0]:
        xpos = np.argmin(sine_lut)
        sine_lut[xpos] = chck_lim[0]
    return np.array(sine_lut, dtype=np.int32)
