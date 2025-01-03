import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from package.digital.dsp import DSP, SettingsDSP, RecommendedSettingsDSP

def load_data(path, filename):
    """Loads a .mat file and returns its content."""
    full_path = os.path.join(path, filename)
    return loadmat(full_path)

def extract_arrays(data, key, num_arrays=61):
    """Extracts a specified number of arrays from the loaded .mat data."""
    return [np.array([arr[i] for arr in data[key]]) for i in range(num_arrays)]

def detect_artifacts(array, threshold_factor=10):
    """Detects artifacts in an array based on standard deviation."""
    mean = np.mean(array)
    std_dev = np.std(array)
    artifacts = np.where(np.abs(array - mean) > threshold_factor * std_dev)[0]
    return artifacts, mean, std_dev

def replace_artifacts_with_spline(array, artifacts, mean, std_dev, threshold_factor=10):
    clean_array = array.copy()
    valid_indices = np.setdiff1d(np.arange(len(array)), artifacts)

    if len(valid_indices) < 2:  # Nicht genug Punkte für Interpolation
        raise ValueError("Nicht genug gültige Punkte für Spline-Interpolation.")

    valid_values = array[valid_indices]

    # Spline interpolation
    spline_interpolator = interp1d(valid_indices, valid_values, kind='cubic', fill_value="extrapolate")
    interpolated_values = spline_interpolator(artifacts)

    lower_bound = mean - threshold_factor * std_dev
    upper_bound = mean + threshold_factor * std_dev
    constrained_values = np.clip(interpolated_values, lower_bound, upper_bound)

    clean_array[artifacts] = constrained_values

    return clean_array


def find_connected_ranges(data):
    diffs = np.diff(data)
    breaks = np.where(diffs >= 5)[0]

    ranges = []
    start_index = 0
    for index in breaks:
        ranges.append((data[start_index], data[index]))
        start_index = index + 1

    ranges.append((data[start_index], data[len(data) - 1]))

    return ranges



def plot_signals(signals, titles, std_devs, means, artifact_ranges, indices_range, figsize=(12, 8)):
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


def filter_signal_based_on_threshold(signal, threshold_limit=755, threshold_factor=10):
    if len(signal) == 0:  # If the signal is empty, automatically ignore it
        return False

    std_dev = np.std(signal)
    computed_threshold = threshold_factor * std_dev
    #print(f"Computed Threshold: {std_dev}")

    return computed_threshold <= threshold_limit


def filter_signals_by_percentage(signal, threshold, percentage_limit=30):
    if len(signal) == 0:
        return False  # Leere Signale ignorieren

    # Berechnen, wie viele Werte den Threshold überschreiten
    values_above_threshold = np.sum(signal > threshold)

    # Prozentsatz berechnen
    percentage_above = (values_above_threshold / len(signal)) * 100

    return percentage_above < percentage_limit


# Main script
if __name__ == "__main__":
    settings = SettingsDSP(
        gain=1,
        fs=25e3,
        n_order=2,
        f_filt=[100, 10000],
        type='iir',
        f_type='butter',
        b_type='bandpass',
        t_dly=0
    )

    plot_counter = 0
    percentage_counter = -1
    threshold_counter = 0
    dsp_instance = DSP(settings)
    dsp_instance.use_filtfilt = True
    path = r"C:\\Users\\jo-di\\Documents\\Masterarbeit\\Rohdaten"
    filename = "A1R1a_elec_stim_50biphasic_400us0001"
    key = "A1R1a_elec_stim_50biphasic_400us0001"

    data = load_data(path, filename)
    result_arrays = extract_arrays(data, key)
    threshold = 150  # Beispiel-Schwellenwert
    percentage_limit = 10 #Beispiel-Prozentwert
    percent_array = []
    threshold_array = []

    for i in range(2):
        original_signal = result_arrays[i]


        if not filter_signals_by_percentage(original_signal, threshold, percentage_limit):
            #print(f"Signal {i} wird ignoriert, da mehr als {threshold_limit}% der Werte über dem Threshold liegen.")
            percentage_counter += 1
            percent_array.append(i)
            continue

        if not filter_signal_based_on_threshold(original_signal, threshold):
            #print(f"Signal {i} wird ignoriert, weil der allgemeine Threshold zu hoch ist.")
            threshold_counter += 1
            threshold_array.append(i)
            continue
        else:
            plot_counter += 1

        filtered_signal = dsp_instance.filter(original_signal)
        original_artifacts, original_mean, original_std_dev = detect_artifacts(original_signal)
        filtered_artifacts, filtered_mean, filtered_std_dev = detect_artifacts(filtered_signal)

        cleaned_original_signal = replace_artifacts_with_spline(
            original_signal,
            original_artifacts,
            original_mean,
            original_std_dev,
            threshold_factor=10
        )
        cleaned_filtered_signal = replace_artifacts_with_spline(
            filtered_signal,
            filtered_artifacts,
            filtered_mean,
            filtered_std_dev,
            threshold_factor=10
        )

        signals = [original_signal, filtered_signal, cleaned_original_signal, cleaned_filtered_signal]
        titles = [
            f"Original Signal {i}",
            "Filtered Signal",
            "Original Signal with Artifacts Replaced",
            "Filtered Signal with Artifacts Replaced"
        ]
        std_devs = [original_std_dev, filtered_std_dev, original_std_dev, filtered_std_dev]
        means = [original_mean, filtered_mean, original_mean, filtered_mean]

        # Plot signals
        for ranges in find_connected_ranges(original_artifacts):
            x = ranges[0]
            y = ranges[1]
            indices_range = (x - 10, y + 10)
            plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)

        indices_range = (None, None)
        plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)
        #print("Filtered Artifacts Indices:", original_artifacts)
        #print("Connected Ranges:", find_connected_ranges(original_artifacts))
        #print(f"Signal: {i}: Stdw:", original_std_dev)
    print(plot_counter, percentage_counter, threshold_counter)
    print(percent_array)
    print(threshold_array)
