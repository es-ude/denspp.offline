import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src_neuro.sda.sda_pipeline import Pipeline_Digital
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA


class SDAPipeline:
    def __init__(self, data_subdir="data", output_subdir="output", data_file_name="signal_dictionary.npy"):
        """
        Initialisiert die SDA Pipeline mit Standardverzeichnissen und Dateinamen.
        """
        self.artifact_indices = []
        self.project_base_dir = Path(__file__).resolve().parent.parent
        self.data_dir_path = self.project_base_dir / data_subdir
        self.output_dir_path = self.project_base_dir / output_subdir
        self.file_path = self.data_dir_path / data_file_name
        self.settings_sda = SettingsSDA(
            fs=25e3,
            dx_sda=[1],
            mode_align=1,
            t_frame_lgth=1.6e-3,
            t_frame_start=0.4e-3,
            dt_offset=[0.1e-3, 0.1e-3],
            t_dly=0.3e-3,
            window_size=7,
            thr_gain=1.0,
            thr_min_value=30
        )
        self.spike_detector = SpikeDetection(self.settings_sda)
        self.signal_dict = {}
        self.spike_indices_list = []
        self.aligned_frames_list = []


    def get_path(self):
        """
        Überprüft und gibt die Daten- und Ausgabepfade zurück.
        """
        if not self.file_path.exists():
            print(f"Die Datei existiert NICHT: {self.file_path}")
        if not self.output_dir_path.exists():
            print(f"Das Ausgabeverzeichnis existiert NICHT: {self.output_dir_path}")
        return self.file_path, self.output_dir_path

    def load_file_as_dict(self):
        """
        Lädt die `.npy`-Datei und gibt sie als Dictionary zurück.
        """
        if not self.file_path.exists():
            print(f"Fehler: Datei wurde nicht gefunden: {self.file_path}")

        try:
            self.signal_dict = np.load(self.file_path, allow_pickle=True).item()
            print(f".npy-Datei erfolgreich als Dictionary geladen: {self.file_path}")

        except Exception as e:
            print(f"Fehler beim Laden der Datei: {self.file_path}. Details: {e}")


    def process_dictionary(self):
        """
        Verarbeitet das geladene Dictionary und extrahiert Artifact-Indizes.
        """
        if not isinstance(self.signal_dict, dict):
            print("Fehler: Geladene Daten sind kein Dictionary!")
            return

        print(f"Keys im geladenen Dictionary: {list(self.signal_dict.keys())}")
        details = self.signal_dict.get("details")
        if details and isinstance(details, dict):
            for signal_key, signal_data in details.items():
                print(f"\nVerarbeite '{signal_key}':")
                timestamps = signal_data.get("timestamps")
                if timestamps:
                    artifacts = timestamps.get("artifacts")
                    if artifacts:
                        indices = artifacts.get("indices")
                        print(f"  - Artifact Indices: {indices}")
                        self.artifact_indices.append(indices)
                    else:
                        print("  - Keine 'artifacts' gefunden.")
                else:
                    print("  - Keine 'timestamps' gefunden.")
        else:
            print("Keine Details im Dictionary gefunden.")


    def compare_indices_in_loop(self, spike_list, artifact_list):
        """
        Vergleicht zwei Listen (Spike- und Artifact-Indizes) auf Überschneidungen.
        """
        common_indices_list = []
        if len(spike_list) != len(artifact_list):
            raise ValueError("Die Arrays haben unterschiedliche Längen und können nicht verglichen werden.")

        for i in range(len(spike_list)):
            spikes = spike_list[i]
            artifacts = artifact_list[i]
            common_indices = np.intersect1d(spikes, artifacts)
            common_indices_list.append(common_indices)

            if common_indices.size > 0:
                print(f"Signal {i + 1}: Es gibt {len(common_indices)} Überschneidungen: {common_indices}")
            else:
                print(f"Signal {i + 1}: Keine Überschneidungen gefunden.")

        print("Vergleich der Arrays abgeschlossen.")
        return common_indices_list

    def time_delay(self, uin):
        """
        Wendet eine Verzögerung auf das Eingangssignal an.
        """
        t_dly = self.settings_sda.t_dly
        set_delay = round(t_dly * self.settings_sda.fs)
        mat = np.zeros(shape=(set_delay,), dtype=float)
        uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
        return uout

    def sda_neo(self, xin):
        """
        Nichtlinearer Energieoperator (NEO) auf das Eingangssignal anwenden.
        """
        ksda0 = self.settings_sda.dx_sda[0]
        x_neo0 = np.floor(xin[ksda0:-ksda0] ** 2 - xin[:-2 * ksda0] * xin[2 * ksda0:])
        x_neo = np.concatenate([x_neo0[:ksda0, ], x_neo0, x_neo0[-ksda0:, ]], axis=None)
        return x_neo

    def sda_smooth(self, xin, window_method="Hamming"):
        """
        Glättet das Eingangssignal mit einem definierten Fenster.
        """
        return self.spike_detector.smoothing_1d(xin, 4 * 1 + 1, window_method)

    def thres_rms(self, xin):
        """
        Berechnet Schwellenwert basierend auf RMS des Signals.
        """
        threshold_gain = self.settings_sda.thr_gain
        window_size = self.settings_sda.window_size
        return threshold_gain * np.sqrt(np.convolve(xin ** 2, np.ones(window_size) / window_size, mode="same"))

    def thres_const(self, xin):
        """
        Wendet einen konstanten Schwellenwert auf das Eingangssignal an.
        """
        return np.zeros(shape=xin.size) + self.settings_sda.thr_min_value

    def process_signals(self, cleaned_signals):
        """
        Verarbeitet bereinigte Signale mithilfe der SDA-Methoden.
        """


        for signal in cleaned_signals:
            x_dly = self.time_delay(signal)
            x_sda = self.sda_neo(x_dly)
            smooth_data = self.sda_smooth(x_sda, "Hamming")
            threshold_data = self.thres_rms(x_sda)
            frames_out0, frames_out1 = self.spike_detector.frame_generation(signal, smooth_data, threshold_data)
            frames_align = frames_out1[0]
            x_pos = frames_out1[1]
            self.aligned_frames_list.append(frames_align)
            self.spike_indices_list.append(x_pos)

    def plot_signals_with_mean(self):
        """
        Plottet Signale aus vorverarbeiteten Frames und hebt den Mittelwert hervor.
        """
        for i, signal_frames in enumerate(self.aligned_frames_list):
            if len(signal_frames) > 0 and i == 1:  # Nur Plot erstellen, wenn Frames vorhanden sind
                all_curves = np.array(signal_frames)

                if all_curves.ndim < 2:
                    print(f"Signal {i}: Nicht genügend Daten, überspringe Plot.")
                    continue

                #mean_curve = np.mean(all_curves, axis=0)

                plt.figure(figsize=(10, 6))
                for j, curve in enumerate(all_curves):
                    if 400 > max(curve) - min(curve) > 80:
                        plt.plot(curve, color="gray", alpha=0.5, linewidth=0.5)
                        #plt.plot(mean_curve, color="black", linewidth=2, label="Mean Curve")
                        plt.title(f"Signal {i + 1} with Spike {j+1}", fontsize=14)
                        plt.xlabel("Sample Index", fontsize=12)
                        plt.ylabel("Signal Amplitude", fontsize=12)
                        plt.grid(alpha=0.3)
                        plt.legend()
                        plt.show()


if __name__ == "__main__":
    pipeline = SDAPipeline()
    file_path, output_dir = pipeline.get_path()
    pipeline.load_file_as_dict()

    if pipeline.signal_dict:
        pipeline.process_dictionary()
        cleaned_signals = pipeline.signal_dict.get("cleaned_signals", [])
        if isinstance(cleaned_signals, list):
            pipeline.process_signals(cleaned_signals)
            pipeline.plot_signals_with_mean()
