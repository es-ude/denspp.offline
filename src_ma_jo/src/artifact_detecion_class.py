import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass

from scipy.interpolate import CubicSpline

from denspp.offline.digital.dsp import SettingsFilter, DSP
from src_ma_jo.src.data_handler_artifacts import load_data, extract_arrays


@dataclass
class ArtifactConfig:
    """
    A dataclass to define and store the configuration parameters for artifact detection.

    Attributes:
        path (str): The directory path where the data files are located.
        filename (str): The name of the file to be loaded.
        plot_flag (bool): Flag to enable or disable plotting of results. Default is True.
        apply_filter (bool): Flag to specify whether to apply filtering. Default is False.
        percentage_limit (float): Percentage limit for signal filtering. Default is 8.
        threshold_limit (float): Threshold limit for signals. Default is 600.
        threshold_factor (float): Multiplication factor for threshold calculation. Default is 10.
        rmse_limit (float): RMSE limit for exponential fitting. Default is 20.
        r_squared_limit (float): R-squared limit for exponential fitting. Default is 0.9.
        num_for_spline_helping_points (int): Number of points used for spline interpolation. Default is 10.
    """

    path: str
    filename: str
    plot_flag: bool = True                  #not implemented yet
    apply_filter: bool = True
    percentage_limit: float = 8
    threshold_limit: float = 400
    threshold_factor: float = 10
    rmse_limit: float = 20
    r_squared_limit: float = 0.9
    num_for_spline_helping_points = 10


class ArtifactDetection:
    """
    A class for detecting and processing artifacts in signals.

    Args:
        filter_settings (SettingsFilter): Configuration for filtering parameters.
        config (ArtifactConfig): Configuration dataclass defining thresholds and other parameters.

    Attributes:
        dsp_instance (DSP): An instance of the digital signal processing class.
        config (ArtifactConfig): A reference to the provided artifact detection configuration.
        time (list): Loaded time array from the data.
        signals (list): Extracted signals from the file.
        add_on (list): Additional arrays extracted, when available.
        artifacts (list): Detected artifacts ranges in signals.
        filtered_signals (list): Signals that have been processed and filtered.
        smoothed_signals (list): Signals after smoothing (if applicable).
        signal_to_process (array): Current signal processed in a pipeline.
        filtered_signal_to_process (array): Current signal being filtered in a pipeline.
        filtered_signals_by_percentage_counter (int): Number of signals filtered by percentage threshold.
        filtered_signals_by_threshold_counter (int): Number of signals filtered by threshold alone.
    """
    def __init__(self, filter_settings: SettingsFilter, config: ArtifactConfig):
        self.dsp_instance = DSP(filter_settings)
        self.config = config

        # Statusvariablen
        self.time = None
        self.signals = None
        self.all_artifacts = []
        self.add_on = None
        self.artifacts = None
        self.filtered_signals = None
        self.smoothed_signals = None
        self.signal_to_process = None
        self.filtered_signal_to_process = None

        self.filtered_signals_by_percentage_counter = 0
        self.filtered_signals_by_threshold_counter = 0

    def load_and_extract_data(self):
        """
        Loads data from the specified path and extracts relevant arrays.

        Raises:
            Exception: If there is an error loading or extracting the data.
        """
        try:
            data = load_data(self.config.path, self.config.filename)
            result_arrays = extract_arrays(data, self.config.filename)

            self.time = result_arrays[0]
            self.signals = result_arrays[1:60]
            if len(result_arrays) > 61:
                self.add_on = result_arrays[-1]

            print(f"Daten erfolgreich geladen und extrahiert: {len(self.signals)+1} Signale gefunden.")

        except Exception as e:
            print(f"Fehler beim Laden oder Extrahieren der Daten: {e}")
            raise

    def filter_and_replace_signals(self):
        """
        Filters signals based on percentage and threshold limits and replaces invalid signals with zeros.
        """
        processed_signals = []

        for signal in self.signals:
            if not self.filtered_signals_by_percentage(signal=signal):
                self.filtered_signals_by_percentage_counter += 1
                processed_signals.append(np.zeros_like(signal))
            elif not self.filter_signal_based_on_threshold(signal=signal):
                self.filtered_signals_by_threshold_counter += 1
                processed_signals.append(np.zeros_like(signal))
            else:
                processed_signals.append(signal)
        self.filtered_signals = processed_signals
        print(f"{len(self.filtered_signals)} Signale erfolgreich gefiltert und verarbeitet.")

    def filtered_signals_by_percentage(self, signal):
        """
        Checks whether the signal meets the percentage threshold filter.

        Args:
            signal (numpy.ndarray): The signal array to process.

        Returns:
            bool: True if the signal passes the percentage threshold, else False.
        """
        if len(signal) == 0:
            return False

        values_above_threshold = np.sum(abs(signal) > self.config.threshold_limit)
        percentage_above = (values_above_threshold / len(signal)) * 100

        return percentage_above < self.config.percentage_limit

    def filter_signal_based_on_threshold(self, signal):
        """
        Filters the signal based on calculated standard deviation and thresholds.

        Args:
            signal (numpy.ndarray): The signal array to process.

        Returns:
            bool: True if the signal passes the threshold filter, else False.
        """
        if len(signal) == 0:
            return False

        std_dev = np.std(signal)
        computed_threshold = self.config.threshold_factor * std_dev

        return computed_threshold <= self.config.threshold_limit

    def process_all_signals(self):
        """
        Processes all signals in `self.filtered_signals`.
        """
        for i, self.signal_to_process in enumerate(self.filtered_signals):
            self.process_single_signal(i)
            self.signals[i] = self.signal_to_process
            self.all_artifacts.append(self.artifacts)

    def process_single_signal(self, index):
        """
        Processes a single signal.

        Args:
            index (int): Index of the signal to process.
        """
        if len(self.signal_to_process) == 0 or not np.any(self.signal_to_process):
            print(f"Signal {index + 1} is empty. Skipping process.")
            return

        mean, std_dev = self.detect_artifacts()
        self.process_artifact_ranges()

    def get_filtered_signal(self):
        """
        Retrieves the filtered signal based on the configuration setting.

        Returns:
            numpy.ndarray: The filtered signal or original signal if filtering is not applied.
        """
        all_filtered_signals = []
        if self.config.apply_filter:
            for signal in self.signals:
                filtered_signal = self.dsp_instance.filter(signal)
                all_filtered_signals.append(filtered_signal)
        else:
            pass
        self.signals = all_filtered_signals


    def detect_artifacts(self):
        """
        Detects artifacts in the current `filtered_signal_to_process`.

        Returns:
            tuple: A tuple consisting of the mean and standard deviation of the signal.

        Raises:
            ValueError: If no artifacts are found in the signal.
        """
        mean = np.mean(self.signal_to_process)
        std_dev = np.std(self.signal_to_process)
        self.artifacts = np.where(np.abs(self.signal_to_process - mean) > 7 * std_dev)[0]
        try:
            if len(self.artifacts) == 0:
                raise ValueError("Keine Artefakte im aktuellen Signal gefunden.")
        except ValueError as e:
            print(f"Fehler: {e}")
            return mean, std_dev
        return mean, std_dev

    def process_artifact_ranges(self):
        """
        Processes ranges marked as artifacts and attempts to correct or replace them.
        """
        if len(self.artifacts) == 0:
            print("Keine Artefakte gefunden.")
            return

        for ranges in self.find_connected_ranges():
            x, y = ranges
            start, end = max(0, x - 10), min(len(self.signal_to_process) - 1, y + 10)
            self.signal_to_process[start:start+20] = np.mean(self.signal_to_process[start:start+20])
            self.replace_artifacts_with_spline([ranges])

    def find_connected_ranges(self):
        """
        Locates connected artifact ranges in the signal.

        Returns:
            list: A list of ranges, where each range contains start and end indices for artifacts.
        """
        diffs = np.diff(self.artifacts)
        breaks = np.where(diffs >= 100)[0]

        ranges = []
        start_index = 0
        for index in breaks:
            ranges.append((self.artifacts[start_index], self.artifacts[index]))
            start_index = index + 1
        ranges.append((self.artifacts[start_index], self.artifacts[-1]))

        return ranges

    def replace_artifacts_with_spline(self, ranges):
        """
        Uses cubic spline interpolation to replace the signal in artifact ranges.

        Args:
            ranges (list): List of artifact ranges to process.
        """
        range_list = list(range(ranges[0][0], ranges[0][1] + 1))
        start_index = max(0, len(range_list) - 20)
        start = range_list[start_index]
        end = range_list[-1]
        exp_data_segment = self.signal_to_process[start:end]
        fitted_curve, rmse, r_squared = self.fit_exponential(exp_data_segment)
        rmse_check = rmse < self.config.rmse_limit
        r2_check = r_squared > self.config.r_squared_limit

        if self.is_saturation(self.signal_to_process[range_list]):
            if rmse_check and r2_check:
                self.signal_to_process[start:end] -= fitted_curve
                self.do_spline_interpolation(range_list[:start_index + 1])
            else:
                self.do_spline_interpolation(range_list)
        else:
            print("Artifact not in saturation region. No spline interpolation performed.")

    def fit_exponential(self, data_segment):
        """
        Fits an exponential function to the given data segment.

        Args:
            data_segment (numpy.ndarray): Signal segment for exponential fitting.

        Returns:
            tuple: Fitted curve, RMSE (float), and R-squared value (float).
        """
        # Validate input
        if len(data_segment) == 0:
            print("Warning: Data segment is empty. Returning defaults.")
            return np.zeros_like(data_segment), float('inf'), 0.0

        if len(data_segment) < 3:  # Ensure there are enough points to fit 3 parameters
            print("Warning: Data segment too short for curve fitting. Returning defaults.")
            return np.zeros_like(data_segment), float('inf'), 0.0

        x = np.arange(len(data_segment))

        try:
            # Attempt to fit the curve
            popt, _ = curve_fit(self.exponential_func, x, data_segment, maxfev=10000, p0=(1, -0.1, 0))
            fitted_curve = self.exponential_func(x, *popt)
        except (RuntimeError, ValueError) as e:
            # Handle failure in curve fitting
            print(f"Curve fitting failed: {e}. Returning defaults.")
            fitted_curve = np.full_like(data_segment, np.mean(data_segment))
            rmse = np.sqrt(np.mean((data_segment - fitted_curve) ** 2))
            return fitted_curve, rmse, 0.0

        # Calculate residuals, RMSE, and R-squared
        residuals = data_segment - fitted_curve
        rmse = np.sqrt(np.mean(residuals ** 2))
        ss_total = np.sum((data_segment - np.mean(data_segment)) ** 2)
        ss_residual = np.sum(residuals ** 2)
        r_squared = 1 - (ss_residual / ss_total if ss_total != 0 else 0)

        return fitted_curve, rmse, r_squared

    def exponential_func(self, x, a, b, c):
        return a * np.exp(b * x) + c

    def sigmoid(self, x, L, k, x0, d):
        return L / (1 + np.exp(-k * (x - x0))) + d

    def do_spline_interpolation(self, artifact_range):

        index_array = np.arange(len(artifact_range))
        start = self.signal_to_process[artifact_range[0]]
        end = self.signal_to_process[artifact_range[-1]]
        indices = np.linspace(0, len(artifact_range), config.num_for_spline_helping_points, dtype=int)
        y = np.linspace(start, end, config.num_for_spline_helping_points, dtype=int)

        spline = CubicSpline(indices, y, bc_type='natural')
        interpolated_signal = spline(index_array)

        self.signal_to_process[artifact_range] = interpolated_signal

    def create_signal_dictionary(self):
        amplitude = self.extract_amplitude_from_filename()
        pass
    def extract_amplitude_from_filename(self):
        try:
            parts = config.filename.split('_')
            extracted_value = parts[3]
            return extracted_value
        except (IndexError, ValueError) as e:
            raise ValueError(f"Fehler beim Extrahieren der Amplitude aus dem Dateinamen '{filename}': {e}")


    @staticmethod
    def is_saturation(signal, flat_length=7):
        diffs = np.diff(signal[9::])
        flat = np.concatenate(([False], diffs == 0))
        count = 0
        for val in flat:
            if val:
                count += 1
                if count >= flat_length:
                    return True
            else:
                count = 0
        return False


if __name__ == "__main__":
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

    config = ArtifactConfig(
        path=r"C:\Users\jo-di\Documents\Masterarbeit\Rohdaten",
        filename="A1R1a_elec_stim_50biphasic_400us0001"
    )

    ad = ArtifactDetection(settings, config)
    ad.load_and_extract_data()
    ad.filter_and_replace_signals()
    ad.process_all_signals()
    ad.get_filtered_signal()

