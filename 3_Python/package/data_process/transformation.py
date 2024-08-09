import numpy as np


def do_fft(y: np.ndarray, fs: float) -> [np.ndarray, np.ndarray]:
    """Performing the Discrete Fast Fourier Transformation
    input:
    y   - transient input signal
    fs  - sampling rate

    output:
    freq    - Frequency
    Y       - Discrete output
    """
    window = np.hanning(y.size)
    fft_in = window * y
    # ---
    N = y.size // 2
    fft_out = 2 / N * np.abs(np.fft.fft(fft_in))
    fft_out[0] = fft_out[0] / 2
    freq = fs * np.fft.fftfreq(fft_out.size)

    # Taking positive range
    xsel = np.where(freq >= 0)
    fft_out = fft_out[xsel]
    freq = freq[xsel]

    return freq, fft_out
