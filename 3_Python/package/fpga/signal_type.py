import numpy as np


def generation_sinusoidal_waveform(bitsize_lut: int, f_rpt: float, f_sine: float, do_optimized=False) -> np.ndarray:
    """Generating the sinusoidal waveform for building LUT file
    Args:
        bitsize_lut:    Used quantization level for generating sinusoidal waveform LUT
        f_rpt:          Frequency of the timer interrupt
        f_sine:         Target frequency of the sinusoidal waveform at output
        do_optimized:   Decision if LUT resources should be minimized [only quarter and mirroring]
    Return:
        Numpy array with waveform [np.int32]
    """
    reduced_samples = 1.0 if not do_optimized else 0.25
    num_lutsine = reduced_samples * f_rpt / f_sine

    # Generating sine waveform as template
    x0 = np.linspace(0, reduced_samples * 2 * np.pi, int(num_lutsine))
    offset = 0 if do_optimized else 1
    sine_lut = (2 ** (bitsize_lut - 1) * (offset + np.sin(x0)))

    # Limitations to output range
    if sine_lut.max() > (2 ** (bitsize_lut - 1) - 1):
        xpos = np.argmax(sine_lut)
        sine_lut[xpos] = (2 ** (bitsize_lut - 1) - 1)
    if sine_lut.min() < 0:
        xpos = np.argmin(sine_lut)
        sine_lut[xpos] = 0
    return np.array(sine_lut, dtype=np.int32)
