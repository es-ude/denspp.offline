import numpy as np

from scipy.optimize import curve_fit
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

from denspp.offline.digital.dsp import SettingsFilter, DSP
from src_ma_jo.src.artifact_detection import exponential_func
from src_ma_jo.src.data_handler_artifacts import load_data, extract_arrays


@dataclass
class ArtifactConfig:
    path: str
    filename: str
    plot_flag: bool = True
    apply_filter: bool = False
    percentage_limit: float = 8
    threshold_limit: float = 600
    threshold_factor: float = 10


class ArtifactDetection:
    def __init__(self, filter_settings: SettingsFilter, config: ArtifactConfig):
        self.dsp_instance = DSP(filter_settings)
        self.config = config

        # Statusvariablen
        self.time = None
        self.signals = None
        self.add_on = None
        self.artifacts = None
        self.filtered_signals = None
        self.smoothed_signals = None

        self.filtered_signals_by_percentage_counter = 0
        self.filtered_signals_by_threshold_counter = 0

    def load_and_extract_data(self):
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
        if len(signal) == 0:
            return False

        values_above_threshold = np.sum(abs(signal) > self.config.threshold_limit)
        percentage_above = (values_above_threshold / len(signal)) * 100

        return percentage_above < self.config.percentage_limit

    def filter_signal_based_on_threshold(self, signal):
        if len(signal) == 0:
            return False

        std_dev = np.std(signal)
        computed_threshold = self.config.threshold_factor * std_dev

        return computed_threshold <= self.config.threshold_limit

    def process_all_signals(self):
        for i, signal in enumerate(self.filtered_signals):
            self.process_single_signal(signal, i)

    def process_single_signal(self, signal, index):
        if len(signal) == 0 or not np.any(signal):
            print(f"Signal {index + 1} is empty. Skipping process.")
            return

        filtered = self.get_filtered_signal(signal)
        self.filtered_signals[index] = filtered
        mean, std_dev = self.detect_artifacts(filtered)
        self.process_artifact_ranges(filtered)

    def get_filtered_signal(self, signal):
        return self.dsp_instance.filter(signal) if self.config.apply_filter else signal

    def detect_artifacts(self, signal_to_detect_from):
        mean = np.mean(signal_to_detect_from)
        std_dev = np.std(signal_to_detect_from)
        self.artifacts = np.where(np.abs(signal_to_detect_from - mean) > self.config.threshold_factor * std_dev)[0]
        try:
            if len(self.artifacts) == 0:
                raise ValueError("Keine Artefakte im aktuellen Signal gefunden.")
        except ValueError as e:
            print(f"Fehler: {e}")
            return mean, std_dev
        return mean, std_dev

    def process_artifact_ranges(self, signal):
        if len(self.artifacts) == 0:
            print("Keine Artefakte gefunden.")
            return signal

        for ranges in self.find_connected_ranges():
            x, y = ranges
            start, end = max(0, x - 10), min(len(signal) - 1, y + 10)
            signal[start:start+20] = np.mean(signal[start:start+20])
            self.replace_artifacts_with_spline([ranges], signal)

    def find_connected_ranges(self):
        diffs = np.diff(self.artifacts)
        breaks = np.where(diffs >= 100)[0]

        ranges = []
        start_index = 0
        for index in breaks:
            ranges.append((self.artifacts[start_index], self.artifacts[index]))
            start_index = index + 1
        ranges.append((self.artifacts[start_index], self.artifacts[-1]))

        return ranges

    def replace_artifacts_with_spline(self, ranges, signal):
        for artifact_range in ranges:
            range_list = list(range(artifact_range[0], artifact_range[1] + 1))
            start = artifact_range[1] - 20
            end = artifact_range[1]
            exp_data_segment = signal[start:end]
            fitted_curve, rmse, r_squared = self.fit_exponential(exp_data_segment)
            rmse_check = rmse < 20
            r2_check = r_squared > 0.9


            if rmse_check and r2_check:
                self.do_spline_interpolation(range_list[0:-20], signal[range_list[0:-20]])
            else:
                signal = self.replace_artifacts_with_sigmoid(range_list, signal[range_list])

    def fit_exponential(self, data_segment):
        x = np.arange(len(data_segment))
        try:
            popt, _ = curve_fit(self.exponential_func, x, data_segment, maxfev=10000, p0=(1, -0.1, 0))
            fitted_curve = exponential_func(x, *popt)
            residuals = data_segment - fitted_curve
        except RuntimeError:
            popt = (0, 0, np.mean(data_segment))
            fitted_curve = np.full_like(data_segment, np.mean(data_segment))
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

    def do_spline_interpolation(self, artifact_range, signal):
        if self.is_saturation(signal[20::]):
            print("Sättigung erkannt, Interpolation übersprungen.")
            return signal

        validated_indices = np.setdiff1d(np.arange(len(signal)), artifact_range)

        if len(validated_indices) < 4:
            raise ValueError("Nicht genügend gültige Punkte für die Spline-Interpolation.")

        try:
            spline = CubicSpline(validated_indices, signal[validated_indices], bc_type="natural", extrapolate=True)
            smooth_curve = spline(np.arange(len(signal)))
        except Exception as e:
            print(f"Fehler bei der Spline-Interpolation: {e}")
            smooth_curve = signal

        return smooth_curve

    def replace_artifacts_with_sigmoid(self, artifact_range, signal):
        x = np.arange(len(signal))
        valid_indices = np.setdiff1d(x, artifact_range)

        if len(valid_indices) < 5:
            print("Zu wenig Daten für sigmoid Fit.")
            return signal

        x_valid = valid_indices
        y_valid = signal[valid_indices]

        try:
            p0 = [np.max(y_valid) - np.min(y_valid), 1, np.median(x_valid), np.min(y_valid)]
            popt, _ = curve_fit(self.sigmoid, x_valid, y_valid, p0=p0, maxfev=10000)

            smooth_curve = self.sigmoid(x, *popt)
        except Exception as e:
            print(f"Sigmoid-Fit fehlgeschlagen: {e}")
            return signal

        return smooth_curve

    @staticmethod
    def is_saturation(signal, flat_length=10):
        diffs = np.diff(signal)
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
        path=r"C:/Users/jo-di/Documents/Masterarbeit/Rohdaten",
        filename="A1R1a_elec_stim_50biphasic_400us0001"
    )

    ad = ArtifactDetection(settings, config)
    ad.load_and_extract_data()
    ad.filter_and_replace_signals()
    ad.process_all_signals()
