import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src_neuro.sda.sda_pipeline import Pipeline_Digital
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA



class SDAPipeline:
    def __init__(self, data_subdir="data", output_subdir="output"):
        """
        Initialisiert die SDA Pipeline mit Standardverzeichnissen und Dateinamen.
        """
        self.data_file_name = "../data/A1R1a_ASIC_1S_1000_15_artifact_dictionary.npy"
        self.artifact_indices = []
        self.project_base_dir = Path(__file__).resolve().parent.parent
        self.data_dir_path = self.project_base_dir / data_subdir
        self.output_dir_path = self.project_base_dir / output_subdir
        self.file_path = self.data_dir_path / self.data_file_name
        self.settings_sda = SettingsSDA(
            fs=25e3,
            dx_sda=[1],
            mode_align=2,
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


    def compare_indices_in_loop(self):
        """
        Vergleicht zwei Listen (Spike- und Artifact-Indizes) auf Überschneidungen.
        """
        common_indices_list = []
        if len(self.spike_indices_list) != len(self.artifact_indices):
            raise ValueError("Die Arrays haben unterschiedliche Längen und können nicht verglichen werden.")

        for i in range(len(self.spike_indices_list)):
            spikes = self.spike_indices_list[i]
            artifacts = self.artifact_indices[i]
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
            x_sda_neo = self.sda_neo(x_dly)
            x_sda_mteo = self.spike_detector.sda_mteo(x_dly)
            x_sda_ado = self.spike_detector.sda_ado(x_dly)
            x_sda_aso = self.spike_detector.sda_aso(x_dly)
            x_sda_spb = self.spike_detector.sda_spb(x_dly)
            x_sda_eed = self.spike_detector.sda_eed(x_dly)
            smooth_data = self.sda_smooth(x_sda_neo, "Hamming")
            threshold_data = self.thres_rms(x_sda_neo)
            frames_out0, frames_out1 = self.spike_detector.frame_generation(signal, smooth_data, threshold_data)
            frames_align = frames_out1[0]
            x_pos = frames_out1[1]
            self.aligned_frames_list.append(frames_align)
            self.spike_indices_list.append(x_pos)

    def plot_single_spikes(self, signal_index=2, save_eps=False):
        """
        Erstellt für jeden Spikeframe eines bestimmten Kanals (standardmäßig Kanal 3 = Index 2)
        eine eigene wissenschaftliche Abbildung mit korrekter Zeitachse.

        Args:
            signal_index (int): Index des zu plottenden Signals (z. B. 2 für Kanal 3).
            save_eps (bool): Wenn True, wird jede Abbildung als EPS-Datei gespeichert.
        """
        if len(self.aligned_frames_list) <= signal_index:
            print(f"Signalindex {signal_index} ist nicht verfügbar.")
            return

        signal_frames = self.aligned_frames_list[signal_index]
        spike_times = self.spike_indices_list[signal_index]
        time_array = self.signal_dict.get("time")

        if time_array is None:
            print("Zeitachse ('time') nicht im Dictionary gefunden.")
            return

        if len(signal_frames) == 0 or len(spike_times) == 0:
            print(f"Keine Frames oder Spikezeiten für Signal {signal_index + 1} vorhanden.")
            return


        for j, frame in enumerate(signal_frames):
            if 400 > max(frame) - min(frame) > 80 and j==32267:
                plt.figure(figsize=(6, 3))
                plt.plot(frame, color="black", linewidth=1)
                plt.xlabel("Frame Samples", fontsize=12)
                plt.ylabel("Amplitude [µV]", fontsize=12)
                #plt.title(f"Spikeframe {j + 1} – Kanal {signal_index + 1}", fontsize=13)
                plt.grid(alpha=0.3)
                plt.tight_layout()


                filename = f"spikeframe_channel{signal_index + 1}_spike{j + 1}.eps"
                plt.savefig(filename, format="eps", dpi=300)

                plt.show()

    def delete_common_indices(self, common_indices_list):
        """
            Löscht die Werte von common_indices aus `self.spike_indices_list`
            und entfernt die zugehörigen Indizes aus `self.aligned_frames_list`.

            Args:
            common_indices_list (list): Liste von numpy-Arrays mit den gemeinsamen Indizes,
                                         die entfernt werden sollen.
            """
        if not isinstance(common_indices_list, list):
            raise ValueError("common_indices_list muss eine Liste sein.")

        for i, common_indices in enumerate(common_indices_list):
            if i >= len(self.spike_indices_list) or i >= len(self.aligned_frames_list):
                raise IndexError(f"Ungültiger Index {i} in den Listen.")

            if not isinstance(common_indices, np.ndarray):
                raise ValueError("common_indices muss ein numpy-Array sein.")

            # Ermitteln der Indizes, wo Überschneidungen vorliegen
            mask = np.isin(self.spike_indices_list[i], common_indices)

            # Entfernen der gemeinsamen Indizes
            self.spike_indices_list[i] = self.spike_indices_list[i][~mask]

            # Entfernen der Zeilen basierend auf der Maske
            self.aligned_frames_list[i] = self.aligned_frames_list[i][~mask]

    def add_to_signal_dict(self):
        """
        Fügt `aligned_frames_list` und `spike_indices_list` dem signal_dict hinzu.
        """
        if isinstance(self.signal_dict, dict):
            self.signal_dict["aligned_frames_list"] = self.aligned_frames_list
            self.signal_dict["spike_indices_list"] = self.spike_indices_list
            print("aligned_frames_list und spike_indices_list zum signal_dict hinzugefügt.")
        else:
            print("Fehler: signal_dict ist kein Dictionary!")


if __name__ == "__main__":
    pipeline = SDAPipeline()
    file_path, output_dir = pipeline.get_path()
    pipeline.load_file_as_dict()

    if pipeline.signal_dict:
        pipeline.process_dictionary()
        cleaned_signals = pipeline.signal_dict.get("cleaned_signal", [])
        if isinstance(cleaned_signals, list):
            pipeline.process_signals(cleaned_signals)
            common_indices_list = pipeline.compare_indices_in_loop()
            pipeline.delete_common_indices(common_indices_list)
            pipeline.add_to_signal_dict()
            pipeline.plot_single_spikes()
            filename = "_".join(pipeline.data_file_name.rsplit(".", 1)[0].split("_")[:-2]) + "_spike_dictionary.npy"
            np.save(filename, pipeline.signal_dict)
