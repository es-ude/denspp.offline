import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from package.digital.dsp import DSP, SettingsDSP, RecommendedSettingsDSP

def load_data(path, filename):
    """Loads a .mat file and returns its content."""
    full_path = os.path.join(path, filename)
    return loadmat(full_path)

def extract_arrays(data, key, num_arrays=3):
    """Extracts a specified number of arrays from the loaded .mat data."""
    return [np.array([arr[i] for arr in data[key]]) for i in range(num_arrays)]

def detect_artifacts(array, threshold_factor=3):
    """Detects artifacts in an array based on standard deviation."""
    mean = np.mean(array)
    std_dev = np.std(array)
    artifacts = np.where(np.abs(array - mean) > threshold_factor * std_dev)[0]
    return artifacts, mean, std_dev

def replace_artifacts(array, artifacts):
    """Replaces artifact values in an array with zero."""
    clean_array = array.copy()
    clean_array[artifacts] = 0
    return clean_array

def find_connected_ranges(data):
    """Finds Connected Ranges in an array."""
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
    """Plots multiple signals with their corresponding information."""
    plt.figure(figsize=figsize)
    for i, (signal, title, std_dev, mean) in enumerate(zip(signals, titles, std_devs, means), 1):
        plt.subplot(len(signals), 1, i)
        start, end = indices_range
        if indices_range is not None:
            plt.plot(signal[start:end], label=title)
        else:
            plt.plot(signal, label=title)
        plt.axhline(mean + 3 * std_dev, color='r', linestyle=':', label='Mean + 3*StdDev')
        plt.axhline(mean - 3 * std_dev, color='r', linestyle=':', label='Mean - 3*StdDev')
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        #plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf)
    settings = RecommendedSettingsDSP
    dsp_instance = DSP(settings)
    fs = settings.fs
    path = r"C:\\Users\\jo-di\\Documents\\Masterarbeit\\Rohdaten"
    filename = "A1R1a_elec_stim_50biphasic_400us0001"
    key = "A1R1a_elec_stim_50biphasic_400us0001"

    data = load_data(path, filename)
    result_arrays = extract_arrays(data, key)

    original_signal = result_arrays[1]
    filtered_signal = dsp_instance.filter(original_signal)

    original_artifacts, original_mean, original_std_dev = detect_artifacts(original_signal)
    filtered_artifacts, filtered_mean, filtered_std_dev = detect_artifacts(filtered_signal)

    cleaned_original_signal = replace_artifacts(original_signal, original_artifacts)
    cleaned_filtered_signal = replace_artifacts(filtered_signal, filtered_artifacts)

    signals = [filtered_signal, cleaned_filtered_signal, original_signal, cleaned_original_signal]
    titles = [
        "Filtered Signal",
        "Filtered Signal with Artifacts Replaced",
        "Original Signal",
        "Original Signal with Artifacts Replaced"
    ]
    std_devs = [filtered_std_dev, filtered_std_dev, original_std_dev, original_std_dev]
    means = [filtered_mean, filtered_mean, original_mean, original_mean]

    indices_range = (333218-10, 333340)
    plot_signals(signals, titles, std_devs, means, [filtered_artifacts, original_artifacts], indices_range)

    print("Filtered Artifacts Indices:", filtered_artifacts)
    #print("Cleaned Filtered Signal Segment:", cleaned_filtered_signal)

    connected_ranges = find_connected_ranges(filtered_artifacts)

    print("Connected Ranges:", connected_ranges)