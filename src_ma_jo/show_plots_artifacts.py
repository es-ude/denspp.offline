import matplotlib.pyplot as plt
import numpy as np

def plot_std_boxplot(signals):
    std_values = [np.std(signal) for signal in signals[1:] if len(signal) > 0]  # Ignorieren leerer Signale
    plt.figure(figsize=(8, 6))
    plt.boxplot(std_values, vert=True, patch_artist=True)
    plt.title("Boxplot der Standardabweichungen")
    plt.ylabel("Standardabweichung (STDW)")
    plt.xlabel("Signalgruppe")
    plt.show()

def plot_std_histogram(signals, bins=20):
    std_values = [np.std(signal) for signal in signals[1:] if len(signal) > 0]  # Ignoriere erstes Signal
    plt.figure(figsize=(8, 6))
    plt.hist(std_values, bins=bins, edgecolor='black', alpha=0.7)
    plt.title("Histogramm der Standardabweichungen (erstes Signal ignoriert)")
    plt.xlabel("Standardabweichung (STD)")
    plt.ylabel("Häufigkeit")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def plot_exponential_fit(data_segment, fitted_curve, artifact_range, figsize=(8, 6)):
    plt.figure(figsize=figsize)
    plt.plot(data_segment, label='Original Data Segment', color='blue')
    plt.plot(fitted_curve, label='Fitted Exponential Curve', color='orange')
    plt.title(f"Exponential Fit for Artifact Range: {artifact_range}")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_artifact_ranges(signal, filtered_signal, cleaned_signal, cleaned_filtered_signal, titles, std_devs, means, artifact_ranges, indices_range, figsize=(12, 8)):
    signals = [signal, filtered_signal, cleaned_signal, cleaned_filtered_signal]
    plt.figure(figsize=figsize)
    for i, (signal, title, std_dev, mean) in enumerate(zip(signals, titles, std_devs, means), 1):
        plt.subplot(len(signals), 1, i)
        start, end = indices_range
        if indices_range is not None:
            plt.plot(signal[start:end], label=title)

        else:
            plt.plot(signal, label=title)
        plt.axhline(mean + 10 * std_dev, color='r', linestyle=':', label='Mean + 10*StdDev')
        plt.axhline(mean - 10 * std_dev, color='r', linestyle=':', label='Mean - 10*StdDev')
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        #plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

