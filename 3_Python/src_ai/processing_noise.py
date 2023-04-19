import sys
import numpy as np
import matplotlib.pyplot as plt

def gen_noise_frame(no_frames: int, frame_in: np.ndarray, wgndB: [int, int]) -> np.ndarray:
    width = frame_in.size
    frames_noise = np.zeros(shape=(no_frames, width), dtype="double")
    frames_out = np.zeros(shape=(no_frames, width), dtype="double")

    # noise_lvl = np.random.uniform(wgndB[0], wgndB[1])
    # TODO: Scaling des Rauschens noch überarbeiten
    noise_lvl = np.random.uniform(0.5, 2)

    for idx in range(0, no_frames):
        frames_noise[idx, :],_ = gen_noise_awgn(width, noise_lvl, False)
        frames_out[idx, :] = frame_in + frames_noise[idx, :]

    return frames_noise, frames_out

def gen_noise_awgn(size: int, scale: float, img: bool) -> np.ndarray:
    if img:
        noise_out = scale * (np.random.randn(size, ) + 1j * np.random.randn(size, ))/np.sqrt(2)
    else:
        noise_out = scale * np.random.randn(size, )

    noise_eff = np.mean(np.abs(noise_out))
    noise_mean = [np.mean(noise_out), np.std(noise_out)]

    return noise_out, noise_eff

def gen_noise_real(t: np.ndarray, wgndB: int, Fc: int, alpha: float):
    N = t.size
    df = 1/(t[1] - t[0])

    noiseLVL = np.power(10, wgndB/20)
    # --- Generate white noise with frequency components
    Uwhite,_ = gen_noise_awgn(N, noiseLVL, False)
    Uwhite = Uwhite - np.mean(Uwhite)
#
    # --- Adapting the amplitude
    Ywhite = 2 * np.abs(np.fft.fft(Uwhite)) / N
    Ywhite[0] = Ywhite[0] / 2

    freq = df * np.fft.fftfreq(t.shape[-1])
    freq = freq[0:int(np.floor(N / 2))]
    X_Fc = np.argwhere(freq >= Fc)
    X_Fc = X_Fc[0]

    # --- Generate pink noise
    M = 2 * N + (N % 2)

    Ypink = np.fft.fft(np.random.randn(1, M))
    Ypink[0] = Ypink[0] / 2

    # Preparing a vector for 1/f multiplication
    NumUniquePts = np.floor(M / 2).astype("int")
    n = np.arange(1, NumUniquePts, 1)
    n = np.power(n, alpha)

    # multiply the left half of the spectrum so the power spectral density is proportional to
    # the frequency by factor 1/f, i.e.the amplitudes are proportional to 1/sqrt(f)
    Y = Ypink[:, 0:NumUniquePts-1] / n

    # prepare a right half of the spectrum - a copy of the left one, except the DC component
    # and Nyquist frequency - they are unique
    NegHalf = np.fliplr(Y[0:NumUniquePts-1])
    Y = np.concatenate((Y, NegHalf), axis=None)

    # --- Generate pink noise
    WindowSize = 1000
    A = np.convolve(a=Ywhite, v=np.ones(WindowSize), mode="full")
    B = 2 * np.convolve(a=np.abs(Y), v=np.ones(WindowSize), mode="full") / M
    ScaleFactor = A[X_Fc] / B[X_Fc]

    U = np.fft.ifft(ScaleFactor * Y)
    Upink = np.real(U[0:N])
    Upink = Upink - np.mean(Upink)

    # --- Generate output noise
    Unoise = Upink + Uwhite

    return (Unoise, Upink, Uwhite)


if __name__ == "__main__":
    print("Test routine for noise generation")
    fs = 20e3
    t = np.arange(0, 1e6 - 1, 1) / fs
    noise_out, noise_eff = gen_noise_awgn(t.size, 0.5, False)
    noise_mean = np.mean(np.abs(noise_out)) + np.zeros(shape=t.shape)
    print(noise_eff)

    plt.figure()
    plt.plot(t, noise_out, color='k')
    plt.plot(t, noise_mean, color='r')
    plt.show(block = True)

    sys.exit()


    print("Test routine for noise generation")
    fs = 20e3
    t = np.arange(0, 1e6-1, 1) / fs
    noise_out = gen_noise_real(t, -110, 100, 0.6)

    df = 1 / (t[1] - t[0])
    freq = df * np.fft.fftfreq(t.shape[-1])
    idx0 = 0
    idx1 = -1
    scale = 1e6

    plt.figure()
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(t, (scale * noise_out[0]))

    ax2 = plt.subplot(3, 3, 2)
    ax2 = plt.hist(x=(scale * noise_out[0]), bins=100)

    ax3 = plt.subplot(3, 3, 3)
    fft_out = np.abs(np.fft.fft(noise_out[0]))
    ax3 = plt.loglog(freq[idx0:idx1], fft_out[idx0:idx1])
    plt.ylim((1e-4, 1e1))

    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(t, (scale * noise_out[1]))

    ax5 = plt.subplot(3, 3, 5)
    ax5 = plt.hist(x=(scale * noise_out[1]), bins=100)

    ax6 = plt.subplot(3, 3, 6)
    fft_out = np.abs(np.fft.fft(noise_out[1]))
    ax6 = plt.loglog(freq[idx0:idx1], fft_out[idx0:idx1])
    plt.ylim((1e-4, 1e1))

    ax7 = plt.subplot(3, 3, 7)
    ax7 = plt.plot(t, (scale * noise_out[2]))
    plt.xlabel("Time t (s)")

    ax8 = plt.subplot(3, 3, 8)
    ax8 = plt.hist(x=(scale * noise_out[2]), bins=100)
    plt.xlabel("U_n (µV)")

    ax9 = plt.subplot(3, 3, 9)
    fft_out = np.abs(np.fft.fft(noise_out[2]))
    ax9 = plt.loglog(freq[idx0:idx1], fft_out[idx0:idx1])
    plt.ylim((1e-4, 1e1))
    plt.xlabel("Frequency f (Hz)")

    plt.show()
