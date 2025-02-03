import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d, CubicSpline
from denspp.offline.digital.dsp import DSP, SettingsDSP, RecommendedSettingsDSP


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


def replace_artifacts_with_spline_smooth(array, artifacts):
    """
    Ersetzt Artefakte mit glatterer Interpolation (CubicSpline) und gibt
    das gesamte glatte Array oder eine geglättete Kurve zurück.
    """
    # Kopie des Arrays erstellen
    clean_array = array.copy()
    clean_array[artifacts[0:10]] = clean_array[artifacts[0]]

    # Gültige Indizes bestimmen
    valid_indices = np.setdiff1d(np.arange(len(array)), artifacts[:-20])
    if len(valid_indices) < 4:  # Mindestens 4 Punkte für CubicSpline erforderlich
        raise ValueError("Nicht genug gültige Punkte für Interpolation.")

    # Interpolation mit CubicSpline
    spline = CubicSpline(valid_indices, clean_array[valid_indices], bc_type='natural')
    smooth_curve = spline(np.arange(len(clean_array)))
    # Punkte ersetzen
    clean_array[artifacts] = smooth_curve[artifacts]

    return clean_array[artifacts]

def prepare_array_for_spline(original_signal):
    return


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


def filter_signal_based_on_threshold(signal, threshold_limit, threshold_factor=10):
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
    values_above_threshold = np.sum(abs(signal) > threshold)

    # Prozentsatz berechnen
    percentage_above = (values_above_threshold / len(signal)) * 100

    return percentage_above < percentage_limit


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
    percentage_counter = -1     # -1, da Zeit immer als %-Threshold klassifiziert wird
    threshold_counter = 0
    dsp_instance = DSP(settings)

    # Flag zur Steuerung der Filterung
    apply_filter = False  # Setze auf False, falls die Filterung übersprungen werden soll

    #TODO: loop über alle files und abspeichern als neue Arrays

    dsp_instance.use_filtfilt = True
    path = r"C:\\Users\\jo-di\\Documents\\Masterarbeit\\Rohdaten"
    filename = "A1R1a_elec_stim_50biphasic_400us0001"
    key = "A1R1a_elec_stim_50biphasic_400us0001"

    data = load_data(path, filename)
    result_arrays = extract_arrays(data, key)

    threshold = 150  # Beispiel-Schwellenwert
    percentage_limit = 10  # Beispiel-Prozentwert
    percent_array = []
    threshold_array = []
    #TODO: Funktion mit übergabeparameter Signal X verarbeiten
    #TODO: Flag für Plot (mit verschiedenen Levels ggf)
    for i in range(2):
        original_signal = result_arrays[i]

        if not filter_signals_by_percentage(original_signal, threshold, percentage_limit):
            # Signal ignorieren, wenn Prozentsatz zu hoch ist
            percentage_counter += 1
            percent_array.append(i)
            continue

        if not filter_signal_based_on_threshold(original_signal, threshold):
            # Signal ignorieren, wenn Threshold überschritten wird
            threshold_counter += 1
            threshold_array.append(i)
            continue
        else:
            plot_counter += 1

        if apply_filter:
            # Filterung anwenden, wenn das Flag aktiviert ist
            filtered_signal = dsp_instance.filter(original_signal)
            filtered_artifacts, filtered_mean, filtered_std_dev = detect_artifacts(filtered_signal)

            # Bereinigung des gefilterten Signals
            cleaned_filtered_signal = replace_artifacts_with_spline(filtered_signal,
                filtered_artifacts)
        else:
            # Filterung überspringen
            filtered_signal = original_signal
            cleaned_filtered_signal = original_signal

        # Artefakterkennung und Bereinigung des Originalsignals
        original_artifacts, original_mean, original_std_dev = detect_artifacts(original_signal)

        # Signale und Metadaten für die Darstellung

        titles = [
            f"Original Signal {i}",
            "Filtered Signal" if apply_filter else "Original Signal (Filter Skipped)",
            "Original Signal with Artifacts Replaced",
            "Filtered Signal with Artifacts Replaced" if apply_filter else "Original Signal with Artifacts Replaced (Filter Skipped)"
        ]
        std_devs = [original_std_dev, filtered_std_dev if apply_filter else original_std_dev, original_std_dev,
                    filtered_std_dev if apply_filter else original_std_dev]
        means = [original_mean, filtered_mean if apply_filter else original_mean, original_mean,
                 filtered_mean if apply_filter else original_mean]

        # Plot für Artefaktbereiche
        cleaned_signal = original_signal.copy()
        for ranges in find_connected_ranges(original_artifacts):
            x = ranges[0]
            y = ranges[1]
            indices_range = (max(0, x - 10), min(len(original_signal) - 1, y + 10))  # Grenzen einhalten
            start, end = indices_range
            range_indices = list(range(start, end))
            replaced_signal = replace_artifacts_with_spline_smooth(original_signal, range_indices)
            cleaned_signal[start:end] = replaced_signal

            signals = [original_signal, filtered_signal, cleaned_signal, cleaned_filtered_signal]
            plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)

        signals = [original_signal, filtered_signal, cleaned_signal, cleaned_filtered_signal]
        # Optional: Gesamtes Signal anzeigen
        indices_range = (None, None)
        plot_signals(signals, titles, std_devs, means, [original_artifacts], indices_range)

    plot_std_boxplot(result_arrays)
    plot_std_histogram(result_arrays)
