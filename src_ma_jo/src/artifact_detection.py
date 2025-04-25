from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from denspp.offline.digital.dsp import DSP, SettingsFilter
from src_ma_jo.src.data_handler_artifacts import load_data, extract_arrays
from src_ma_jo.src.show_plots_artifacts import *
import numpy as np
from scipy.io import savemat


def save_signal_dictionary_as_mat(signal_dict, filename="signal_dictionary.mat"):
    """
    Speichert ein gegebenes Signal-Dictionary in einer .mat-Datei.

    :param signal_dict: Das zu speichernde Signal-Dictionary
    :param filename: Name der Datei, in der das Dictionary gespeichert wird
    """
    try:
        savemat(filename, signal_dict)
        print(f"Signal-Dictionary wurde erfolgreich als '{filename}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der .mat-Datei: {e}")


def extract_amplitude_from_filename(filename):
    """
    Extracts the amplitude from a given filename.

    :param filename: Filename as a string
    :return: Voltage value as a float
    :raises ValueError: If the amplitude cannot be extracted from the filename
    """
    """
    Extahiert die Amplitude aus dem Dateinamen.

    :param filename: Dateiname als String
    :return: Spannungswert als Float
    """
    try:
        parts = filename.split('_')
        extracted_value = parts[3]
        return extracted_value
    except (IndexError, ValueError) as e:
        raise ValueError(f"Fehler beim Extrahieren der Amplitude aus dem Dateinamen '{filename}': {e}")
    else:
        raise ValueError("Amplitude konnte aus dem Dateinamen nicht extrahiert werden.")


def create_signal_dictionary(filenames, signals, time, artifacts, artifact_indices):
    """
    Creates a dictionary of processed signals.

    :param filenames: List of filenames
    :param signals: List of associated processed signals (arrays)
    :param signal_index: Index of the current signal being processed
    :param artifact_indices: List of artifact indices for each signal
    :return: Nested dictionary with processed signals and metadata
    :return: Nested dictionary with processed signals and metadata
    """

    # Extract amplitude from filename
    amplitude = extract_amplitude_from_filename(filenames)

    # Initialize the dictionary
    processed_signals = {}

    # Assign time (typically the first element in the signals list)
    time = time

    # Assign cleaned signals (everything except the first element in the signals list)
    cleaned_signals = signals

    # Iterate over the cleaned signals and create individual entries
    for idx, signal in enumerate(cleaned_signals):
        processed_signals[f"signal_{idx + 1}"] = {
            "timestamps": {
                "signal": signal.tolist() if hasattr(signal, "tolist") else signal,
                "artifacts": {
                    "indices": artifact_indices[idx],
                    "details": artifacts[idx] if idx < len(artifacts) else []
                }
            }
        }

    # Return the final signal dictionary
    final_dictionary = {
        "time": time.tolist() if hasattr(time, "tolist") else time,
        "cleaned_signals": cleaned_signals,
        "amplitude": amplitude,
        "details": processed_signals
    }

    return final_dictionary


def detect_artifacts(array, threshold_factor=10):
    """
    Detects artifacts in an array based on standard deviation and a threshold factor.

    :param array: Input array to check for artifacts
    :param threshold_factor: Multiplier for the standard deviation to define thresholds
    :return: Tuple containing indices of artifacts, mean, and standard deviation
    """
    """Detects artifacts in an array based on standard deviation."""
    mean = np.mean(array)
    std_dev = np.std(array)
    artifacts = np.where(np.abs(array - mean) > threshold_factor * std_dev)[0]
    try:
        if len(artifacts) == 0:
            raise ValueError("Keine Artefakte im aktuellen Array gefunden.")
    except ValueError as e:
        print(f"Fehler: {e}")
        return np.array([]), mean, std_dev
    return artifacts, mean, std_dev


def exponential_func(x, a, b, c):
    """
    Exponential mathematical function.

    :param x: Input value or array
    :param a: Amplitude parameter
    :param b: Exponent parameter
    :param c: Offset parameter
    :return: Resultant value of the function
    """
    return a * np.exp(b * x) + c

def fit_exponential(data_segment, debug=True):
    """
    Fits an exponential function to a given data segment.

    :param data_segment: Array of data points to fit
    :param debug: If True, additional logs will be printed
    :return: Fitted parameters, curve, residuals, RMSE, and R² value
    """
    x = np.arange(len(data_segment))
    try:
        popt, _ = curve_fit(exponential_func, x, data_segment, maxfev=10000, p0=(1, -0.1, 0))
        fitted_curve = exponential_func(x, *popt)
        residuals = data_segment - fitted_curve
    except RuntimeError:
        popt = (0, 0, np.mean(data_segment))
        fitted_curve = np.full_like(data_segment, np.mean(data_segment))
        residuals = data_segment - fitted_curve
    rmse = np.sqrt(np.mean(residuals**2))
    ss_total = np.sum((data_segment - np.mean(data_segment))**2)
    ss_residual = np.sum(residuals**2)
    r_squared = 1 - (ss_residual / ss_total if ss_total != 0 else 0)
    if debug:
        print("Fit Parameters:", popt)
        print("RMSE:", rmse)
        print("R²:", r_squared)
        residuals = data_segment - fitted_curve
    return popt, fitted_curve, residuals, rmse, r_squared

def compare_std_deviation_exponential(artifact_array, threshold):
    """
    Compares the standard deviation between an exponential fit and the last 20 values
    of the artifact array below a defined threshold.

    :param artifact_array: Array of artifact values to analyze
    :param threshold: Threshold for acceptable standard deviation
    :return: True if deviation is below threshold, otherwise False
    :raises ValueError: If the artifact array contains fewer than 20 values
    """
    """
    Vergleicht die Standardabweichung zwischen einer Exponentialfunktion und den letzten 20 Werten
    des Artefakts mit einem gegebenen Schwellenwert.

    :param artifact_array: Array mit Artefaktwerten
    :param threshold: Schwellenwert für die Standardabweichung
    :return: True, wenn die Standardabweichung unter dem Schwellenwert liegt, andernfalls False
    """
    if len(artifact_array) < 20:
        raise ValueError("Das Artefakt-Array enthält weniger als 20 Werte.")

    data_segment = artifact_array[-20:]

    popt, fitted_curve, residuals, rmse, r_squared = fit_exponential(data_segment, debug=True)
    differences = data_segment - fitted_curve
    std_dev = np.std(differences)
    print(std_dev)

    # Vergleiche mit dem Schwellenwert
    return std_dev <= threshold

def replace_artifacts_with_spline_smooth(array, artifacts, std_threshold=10):
    """
    Replaces artifacts in an array with smoother interpolated (CubicSpline) values.

    :param array: Original signal array
    :param artifacts: Indices of artifact values
    :param std_threshold: Threshold for standard deviation during artifact checking
    :return: Cleaned signal array with artifacts replaced
    :raises ValueError: If there are insufficient valid points for interpolation
    """
    clean_array = array.astype(float).copy()

    for artifact_range in find_connected_ranges(artifacts):
        start = artifact_range[1]-20
        end = artifact_range[1]
        exp_data_segment = clean_array[start:end]
        _, fitted_curve, residuals, rmse, r_squared = fit_exponential(exp_data_segment, debug=False)
        rmse_check = rmse < 20
        r2_check = r_squared > 0.9
        print(rmse_check, r2_check, rmse, r_squared)

        # Überarbeiteter Bereich
        if rmse_check and r2_check:
            plot_exponential_fit(exp_data_segment, fitted_curve, artifact_range)
            x_artifact = np.arange(len(clean_array))[artifact_range[0]:artifact_range[1]]
            extrapolated_values = exponential_func(np.arange(len(x_artifact)), *_)
            clean_array[artifact_range[0]:artifact_range[1]] -= extrapolated_values
        else:
            clean_array[start:end] = np.mean(exp_data_segment)
            print(clean_array[start:end])
            valid_indices = np.setdiff1d(
                np.arange(max(0, artifacts[0] - 10), min(len(array), artifacts[-1] + 10)),
                artifacts
            )
            if len(valid_indices) < 2:
                raise ValueError("Zu wenige Punkte für die CubicSpline-Interpolation.")

            spline = CubicSpline(valid_indices, clean_array[valid_indices], bc_type='natural', extrapolate=True)
            artifact_range = (
                max(0, artifact_range[0]),
                min(len(clean_array), artifact_range[1])
            )
            clean_array[artifact_range[0]:artifact_range[1]] = spline(np.arange(artifact_range[0], artifact_range[1]))

    valid_indices = np.setdiff1d(np.arange(len(array)), artifacts)
    if len(valid_indices) < 4:  # Mindestens 4 Punkte für CubicSpline erforderlich
        raise ValueError("Nicht genug gültige Punkte für Interpolation.")
    else:
        spline = CubicSpline(valid_indices, clean_array[valid_indices], bc_type='natural')
        smooth_curve = spline(np.arange(len(clean_array)))
        clean_array[artifacts] = smooth_curve[artifacts]

    return  clean_array[artifacts]


def process_signal(signal, signal_index, dsp_instance, apply_filter, percentage_limit, threshold, plot_flag=False):
    """
    Processes a single signal by applying filtering, detecting artifacts, and replacing them.

    :param signal: Signal array to process
    :param signal_index: Index of the signal in the array
    :param dsp_instance: DSP instance to apply filtering
    :param apply_filter: Whether to apply a filter to the signal
    :param percentage_limit: Limit for percentage-based filtering
    :param threshold: Threshold for artifact detection
    :param plot_flag: If True, plots may be generated
    :return: Dictionary of processing results
    """
    if len(signal) == 0 or any(signal) == 0:
        print(f"Signal {signal_index} is empty. Skipping process.")
        return {
            "plot_counter": 0,
            "percentage_counter": 0,
            "threshold_counter": 0,
            "percent_array": [],
            "threshold_array": [],
        }
    cleaned_signal = signal.copy()
    plot_counter = 0
    percentage_counter = 0
    threshold_counter = 0
    percent_array = []
    threshold_array = []

    if not filter_signals_by_percentage(signal, threshold, percentage_limit):
        percentage_counter += 1
        percent_array.append(signal_index)
        return {
            "plot_counter": plot_counter,
            "percentage_counter": percentage_counter,
            "threshold_counter": threshold_counter,
            "percent_array": percent_array,
            "threshold_array": threshold_array,
        }

    if not filter_signal_based_on_threshold(signal, threshold):
        threshold_counter += 1
        threshold_array.append(signal_index)
        return {
            "plot_counter": plot_counter,

            "threshold_counter": threshold_counter,
            "percent_array": percent_array,
            "threshold_array": threshold_array,
        }
    else:
        plot_counter += 1

    if apply_filter or apply_spline_modification:
        filtered_signal = dsp_instance.filter(signal)
        filtered_artifacts, filtered_mean, filtered_std_dev = detect_artifacts(filtered_signal, threshold_factor=15)
        cleaned_filtered_signal = replace_artifacts_with_spline_smooth(
            filtered_signal, filtered_artifacts
        )
    else:
        filtered_signal = signal
        cleaned_filtered_signal = signal

    original_artifacts, original_mean, original_std_dev = detect_artifacts(signal)

    titles = [
        f"Original Signal {signal_index}",
        "Filtered Signal" if apply_filter else "Original Signal (Filter Skipped)",
        "Original Signal with Artifacts Replaced",
        "Filtered Signal with Artifacts Replaced" if apply_filter else "Original Signal with Artifacts Replaced (Filter Skipped)",
    ]
    std_devs = [
        original_std_dev,
        filtered_std_dev if apply_filter else original_std_dev,
        original_std_dev,
        filtered_std_dev if apply_filter else original_std_dev,
    ]
    means = [
        original_mean,
        filtered_mean if apply_filter else original_mean,
        original_mean,
        filtered_mean if apply_filter else original_mean,
    ]

    cleaned_signal = process_artifact_ranges(
        signal, filtered_signal, cleaned_signal, cleaned_filtered_signal,
        original_artifacts, titles, std_devs, means, plot_flag)
    if plot_flag:
        plot_artifact_ranges(
            signal, filtered_signal, cleaned_signal, cleaned_filtered_signal,
            titles, std_devs, means, original_artifacts, (None, None)
        )

    return {
        "plot_counter": plot_counter,
        "percentage_counter": percentage_counter,
        "threshold_counter": threshold_counter,
        "percent_array": percent_array,
        "threshold_array": threshold_array,
    }


def process_artifact_ranges(signal, filtered_signal, cleaned_signal, cleaned_filtered_signal, original_artifacts, titles, std_devs, means, plot_flag):
    """
    Processes artifact ranges in the signal and applies corrections.

    :param signal: Original unprocessed signal
    :param filtered_signal: Signal after filtering
    :param cleaned_signal: Signal with artifacts replaced
    :param cleaned_filtered_signal: Filtered signal with artifacts replaced
    :param original_artifacts: Detected artifact ranges
    :param titles: Titles for plots being generated
    :param std_devs: Standard deviation values
    :param means: Mean values of different signals
    :param plot_flag: If True, plots the artifact ranges
    :return: Signal with artifact ranges corrected
    """
    #TODO: artefakt merge
    if len(original_artifacts) == 0:
        print("Original artifacts are empty. Returning the signal as is.")
        return signal
    cleaned_signal = signal.copy()
    for ranges in find_connected_ranges(original_artifacts):
        x = ranges[0]
        y = ranges[1]
        indices_range = (max(0, x - 10), min(len(signal) - 1, y + 10))
        start, end = indices_range
        range_indices = list(range(start, end))
        signal[start:start+20] = np.mean(signal[start:start+20])
        replaced_signal = replace_artifacts_with_spline_smooth(signal, range_indices)
        if len(replaced_signal) != (end - start):
            trimmed_length = min(len(replaced_signal), end - start)
            replaced_signal = replaced_signal[:trimmed_length]
        cleaned_signal[start:start+len(replaced_signal)] = replaced_signal
        if plot_flag:
            plot_artifact_ranges(
                signal, filtered_signal, cleaned_signal, cleaned_filtered_signal,
                titles, std_devs, means, original_artifacts, indices_range
            )

    return cleaned_signal

def find_connected_ranges(artifacts_data):
    """
    Groups connected values in the data into ranges.

    :param data: Array of indices or values
    :return: List of tuples representing start and end points of connected ranges
    """
    if len(artifacts_data) == 0:
        return []
    diffs = np.diff(artifacts_data)
    breaks = np.where(diffs >= 100)[0]

    ranges = []
    start_index = 0
    for index in breaks:
        ranges.append((artifacts_data[start_index], artifacts_data[index]))
        start_index = index + 1

    ranges.append((artifacts_data[start_index], artifacts_data[len(artifacts_data) - 1]))

    return ranges

def filter_signal_based_on_threshold(signal, threshold_limit, threshold_factor=15):
    """
    Filters signals based on a dynamic threshold calculated from standard deviation.

    :param signal: Signal array to evaluate
    :param threshold_limit: Maximum allowable threshold
    :param threshold_factor: Multiplier for standard deviation
    :return: True if the calculated threshold is within the limit, otherwise False
    """
    if len(signal) == 0:
        return False

    std_dev = np.std(signal)
    computed_threshold = threshold_factor * std_dev
    print(f"Computed Threshold: {computed_threshold}")
    print(f"std_dev: {std_dev}")

    return computed_threshold <= threshold_limit


def filter_signals_by_percentage(signal, threshold, percentage_limit=30):
    """
    Filters signals based on the percentage of values exceeding a given threshold.

    :param signal: Signal array to evaluate
    :param threshold: Threshold for evaluation
    :param percentage_limit: Maximum allowed percentage of values exceeding the threshold
    :return: True if the percentage is below the limit, otherwise False
    """
    if len(signal) == 0:
        return False

    # Berechnen, wie viele Werte den Threshold überschreiten
    values_above_threshold = np.sum(abs(signal) > threshold)

    # Prozentsatz berechnen
    percentage_above = (values_above_threshold / len(signal)) * 100

    return percentage_above < percentage_limit


def process_signals(result_arrays, dsp_instance, apply_filter, percentage_limit, threshold, plot_flag):
    """
    Processes a list of signal arrays, applying filtering and artifact detection.

    :param result_arrays: List of signal arrays
    :param dsp_instance: DSP instance for filtering
    :param apply_filter: Whether to enable filtering
    :param percentage_limit: Limit for percentage-based filtering
    :param threshold: Threshold for artifact detection
    :param plot_flag: If True, triggers plot generation
    :return: Dictionary of processing results containing counters and indices
    """
    plot_counter = 0
    percentage_counter = -1  # -1, da Zeit immer als %-Threshold klassifiziert wird
    threshold_counter = 0
    percent_array = []
    threshold_array = []
    for i, signal in enumerate(result_arrays):
        result = process_signal(
            signal=signal,
            signal_index=i,
            dsp_instance=dsp_instance,
            apply_filter=apply_filter,
            percentage_limit=percentage_limit,
            threshold=threshold,
            plot_flag=plot_flag
        )
        plot_counter += result["plot_counter"]
        percentage_counter += result["percentage_counter"]
        threshold_counter += result["threshold_counter"]
        percent_array.extend(result["percent_array"])
        threshold_array.extend(result["threshold_array"])
    return {
        "plot_counter": plot_counter,
        "percentage_counter": percentage_counter,
        "threshold_counter": threshold_counter,
        "percent_array": percent_array,
        "threshold_array": threshold_array,
    }
def replace_signals_with_zeros(signals, percentage_limit, threshold):
    """
    Replaces signals that fail both the percentage and threshold filters with zeros.

    :param signals: List of signal arrays
    :param percentage_limit: Limit for percentage-based filtering
    :param threshold: Threshold for artifact detection
    :return: List of processed signals with invalid signals replaced by zeros
    """
    processed_signals = []
    for signal in signals:
        if not filter_signals_by_percentage(signal, threshold, percentage_limit):
            processed_signals.append(np.zeros_like(signal))
        elif not filter_signal_based_on_threshold(signal, threshold):
            processed_signals.append(np.zeros_like(signal))
        else:
            processed_signals.append(signal)
    return processed_signals

def save_signal_dictionary(signal_dict, filename="signal_dictionary.npy"):
    """
    Saves a given signal dictionary to a .npy file.

    :param signal_dict: Signal dictionary to save
    :param filename: Name of the file to save the dictionary
    """
    """
    Speichert ein gegebenes Signal-Dictionary in einer .npy-Datei.

    :param signal_dict: Das zu speichernde Signal-Dictionary
    :param filename: Name der Datei, in der das Dictionary gespeichert wird
    """
    np.save(filename, signal_dict)

# Main script

if __name__ == "__main__":
    processed_signals_list = []
    settings = SettingsFilter(
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
    apply_filter = False     # Setze auf False, falls die Filterung übersprungen werden soll
    apply_spline_modification = True  # Neues Flag für Spline-Modifikation

    dsp_instance.use_filtfilt = True
    path = r"C:/Users/jo-di/Documents/Masterarbeit/Rohdaten"
    filename = "A1R1a_elec_stim_50biphasic_400us0001"

    data = load_data(path, filename)
    result_arrays = extract_arrays(data, filename)
    time = result_arrays[0]

    threshold = 400  # Beispiel-Schwellenwert
    percentage_limit = 8  # Beispiel-Prozentwert
    percent_array = []
    threshold_array = []

    processed_signals = replace_signals_with_zeros(
        signals=result_arrays[1:],  # Zeit wird ausgelassen
        percentage_limit=percentage_limit,
        threshold=threshold
    )

    result = process_signals(
        result_arrays=processed_signals,
        dsp_instance=dsp_instance,
        apply_filter=apply_filter,
        percentage_limit=percentage_limit,
        threshold=threshold,
        plot_flag=False
    )

    artifact_indices = [
        detect_artifacts(signal, threshold_factor=10)[0].tolist() if len(signal) > 0 else []
        for signal in processed_signals
    ]
    signal_dictionary = create_signal_dictionary(filename, processed_signals, time, artifacts=[], artifact_indices=artifact_indices)
    save_signal_dictionary(signal_dictionary, filename +".npy")
    #save_signal_dictionary_as_mat(signal_dictionary, filename + ".mat")
    plot_counter = result["plot_counter"]
    percentage_counter = result["percentage_counter"]
    threshold_counter = result["threshold_counter"]
    percent_array = result["percent_array"]
    threshold_array = result["threshold_array"]
    plot_std_boxplot(result_arrays)
    plot_std_histogram(result_arrays)
    print(plot_counter, percentage_counter, threshold_counter)
    #print(signal_dictionary)
