import numpy as np
import matplotlib.pyplot as plt

# TODO: Werte für Frame-Noise noch anpassen
def frame_noise(no_frames: int, frame_in: np.ndarray, wgndB: [float, float], fs: float) -> [np.ndarray, np.ndarray]:
    """Generation of noisy spike frames with AWGN at noise power [dB]"""
    width = frame_in.size
    frames_noise = np.zeros(shape=(no_frames, width), dtype="double")
    frames_out = np.zeros(shape=(no_frames, width), dtype="double")
    # Calculation
    noise_lvl = np.random.uniform(wgndB[0], wgndB[1])
    for idx in range(0, no_frames):
        frames_noise[idx, :] = noise_awgn(width, fs, noise_lvl)[0]
        frames_out[idx, :] = frame_in + frames_noise[idx, :]
    return frames_noise, frames_out

def noise_awgn(size: int, fs: float, wgndBW: float) -> [np.ndarray, np.ndarray]:
    """Generation of transient noise signal with noise power [dB]
    input:
    wgndB   - spectral noise density [dBW/SQRT(Hz)]
    fs      - sample rate
    size    - no of points
    output:
    noise   - N points of noise signal with spectral noise density of rho
    n_eff   - effective value of output noise
    """
    rho = 10 ** (wgndBW / 10)
    sigma = rho * np.sqrt(fs/2)
    noise = np.random.normal(0, sigma, size)
    noise -= np.mean(noise)
    # Calculation of effective noise
    noise_eff = np.std(noise)
    noise_pp = np.max(noise)-np.min(noise)
    return noise, noise_eff

def noise_flicker(size: int, alpha: float) -> np.ndarray:
    M = 2 * size + (size % 2)
    # --- Thermal noise
    ynoise = np.random.randn(M)
    window = np.hanning(M)
    Ypink = np.fft.fft(window * ynoise)
    Ypink[0] = Ypink[0] / 2
    # --- Flicker spectral component
    n = np.arange(1, size+1, 1)
    n = np.power(n, alpha)
    # --- Generate full spectrum
    Y = Ypink[0:size] / n
    Ypos = Y[np.newaxis, :]
    Yneg = np.fliplr(Ypos)
    Y = np.concatenate((Ypos, Yneg), axis=None)
    # --- Generate pink noise
    U = np.fft.ifft(Y)
    Upink = np.real(U[0:size])
    Upink -= np.mean(Upink)
    # Output
    return Upink

def noise_real(tsize: int, fs: float, wgndBW: float, Fc: float, alpha: float) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Generation of real noise signal with specific noise power [dBW], corner frequency Fc, and slope of 1/f noise"""
    # --- Generate noise components and match
    Uwhite,_ = noise_awgn(tsize, fs, wgndBW)
    Upink = noise_flicker(tsize, alpha)
    # --- Adapting the amplitude
    freq0, Ywhite = do_fft(Uwhite, fs)
    _, Ypink = do_fft(Upink, fs)
    # --- Find corner frequency
    X_Fc = np.argwhere(freq0 >= Fc)
    X_Fc = X_Fc[0]
    Nmean = 100
    YWm = np.convolve(Ywhite, np.ones(Nmean)/Nmean, mode='same')
    YPm = np.convolve(Ypink, np.ones(Nmean)/Nmean, mode='same')
    scalef = YWm[X_Fc] / YPm[X_Fc]
    # --- Generate output noise
    Unoise = scalef * Upink + Uwhite
    noise_eff = np.std(Unoise)
    noise_pp = np.max(Unoise) - np.min(Unoise)
    # print(f"... effective noise voltage of {1e6*noise_eff:.5f} µV")
    return Unoise, noise_eff, noise_pp

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

# -------- TEST ROUTINE ------------
if __name__ == "__main__":
    fs = 2e3
    t = np.arange(0, 2e6, 1) / fs

    # Real Noise
    noise_out, noise_pink, noise_f = noise_real(t.size, fs, -130, 1, 0.9)
    freq1, psd_real = do_fft(noise_out, fs)

    scale = 1e3
    plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    ax4 = plt.subplot(3, 3, 4)
    ax5 = plt.subplot(3, 3, 5)
    ax6 = plt.subplot(3, 3, 6)
    ax7 = plt.subplot(3, 3, 7)
    ax8 = plt.subplot(3, 3, 8)
    ax9 = plt.subplot(3, 3, 9)

    ax1.set_title('Thermal noise')
    ax1.plot(t, scale*noise_f)
    ax4.hist(scale*noise_f, bins=100, density=True)
    ax7.psd(scale*noise_f, Fs=fs)

    ax2.set_title('Flicker noise')
    ax2.plot(t, scale*noise_pink)
    ax5.hist(scale*noise_pink, bins=100, density=True)
    ax8.psd(scale*noise_pink, Fs=fs)

    ax3.set_title('Real noise')
    ax3.plot(t, scale*noise_out)
    ax6.hist(scale*noise_out, bins=100, density=True)
    ax9.loglog(freq1, psd_real)

    plt.tight_layout()
    plt.show()
