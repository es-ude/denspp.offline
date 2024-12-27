import numpy as np
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
from package.digital.dsp import DSP, SettingsDSP, RecommendedSettingsDSP

def load_data(path, filename):
    """Loads a .mat file and returns its content."""
    full_path = os.path.join(path, filename)
    return loadmat(full_path)

def extract_arrays(data, key, num_arrays=10):
    """Extracts a specified number of arrays from the loaded .mat data."""
    return [np.array([arr[i] for arr in data[key]]) for i in range(num_arrays)]

def detect_artifacts(array, threshold_factor=10):
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
        plt.axhline(mean + 10 * std_dev, color='r', linestyle=':', label='Mean + 10*StdDev')
        plt.axhline(mean - 10 * std_dev, color='r', linestyle=':', label='Mean - 10*StdDev')
        plt.title(title)
        plt.xlabel("Index")
        plt.ylabel("Amplitude")
        #plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Main script
if __name__ == "__main__":
    #np.set_printoptions(threshold=np.inf)
    settings =  SettingsDSP(
        gain=1,
        fs=25e3,
        n_order=2,
        f_filt=[1000,10000],
        type='iir',
        f_type='butter',
        b_type='bandpass',
        t_dly=0
    )
    dsp_instance = DSP(settings)
    dsp_instance.use_filtfilt = True
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

    signals = [original_signal, filtered_signal, cleaned_original_signal, cleaned_filtered_signal]
    titles = [
        "Original Signal",
        "Filtered Signal",
        "Original Signal with Artifacts Replaced",
        "Filtered Signal with Artifacts Replaced"
    ]
    std_devs = [original_std_dev, filtered_std_dev, original_std_dev, filtered_std_dev]
    means = [original_mean, filtered_mean, original_mean, filtered_mean]
    for ranges in find_connected_ranges(original_artifacts):
        x = ranges[0]
        y = ranges[1]
        indices_range = (x-10,y+10)
        plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)
    indices_range = (None, None)
    plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)
    print("Filtered Artifacts Indices:", original_artifacts)
    #print("Cleaned Filtered Signal Segment:", cleaned_filtered_signal)

    connected_ranges = find_connected_ranges(original_artifacts)

    print("Connected Ranges:", connected_ranges)