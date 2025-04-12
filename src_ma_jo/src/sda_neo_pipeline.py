from pathlib import Path
import numpy as np
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA


def get_path(data_subdir="data", output_subdir="output", data_file_name="A1R1a_elec_stim_50biphasic_400us0001.npy"):
    """
    Dynamisch generiert Pfade relativ zur Datei-Position des Skripts.

    :param data_subdir: Name des Datenunterordners
    :param output_subdir: Name des Ausgabeunterordners
    :param data_file_name: Name der Zieldatei
    :return: Tuple (Datenpfad, Ausgabeordner)
    """
    base_dir = Path(__file__).resolve().parent.parent

    data_dir = base_dir / data_subdir
    output_dir = base_dir / output_subdir
    data_file = data_dir / data_file_name

    if not data_file.exists():
        print(f"Die Datei existiert NICHT: {data_file}")
    if not output_dir.exists():
        print(f"Das Ausgabeverzeichnis existiert NICHT: {output_dir}")

    return data_file, output_dir


def load_file_as_dict(file_path):
    """
    Lädt eine .npy-Datei, die ein Dictionary enthält.

    :param file_path: Pfad zur .npy-Datei
    :return: Das Dictionary, wenn erfolgreich geladen; None im Fehlerfall
    """
    if not Path(file_path).exists():
        print(f"Fehler: Datei wurde nicht gefunden: {file_path}")
        return None

    try:
        data = np.load(file_path, allow_pickle=True).item()
        print(f".npy-Datei erfolgreich als Dictionary geladen: {file_path}")
        return data
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {file_path}. Details: {e}")
        return None


def process_dictionary(data_dict):
    """
    Iteriert über verschachtelte dictionary-Daten und verarbeitet diese.

    :param data_dict: Das geladene Dictionary
    """
    if not isinstance(data_dict, dict):
        print("Fehler: Geladene Daten sind kein Dictionary!")
        return

    print(f"Keys im geladenen Dictionary: {list(data_dict.keys())}")

    details = data_dict.get("details")
    if details and isinstance(details, dict):
        for signal_key, signal_data in details.items():
            print(f"\nVerarbeite '{signal_key}':")
            timestamps = signal_data.get("timestamps")
            if timestamps:
                artifacts = timestamps.get("artifacts")
                if artifacts:
                    indices = artifacts.get("indices")
                    print(f"  - Artifact Indices: {indices}")
                else:
                    print("  - Keine 'artifacts' gefunden.")
            else:
                print("  - Keine 'timestamps' gefunden.")
    else:
        print("Keine Details im Dictionary gefunden.")


if __name__ == "__main__":
    file_path, output_dir = get_path()
    data_dict = load_file_as_dict(file_path)
    CustomSettingsSDA = SettingsSDA(
        fs=25e3,
        dx_sda=[1],
        mode_align=1,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.1e-3, 0.1e-3],
        t_dly=0.3e-3,
        window_size=7,
        thr_gain=1.0,
        thr_min_value=100.0
    )
    spike_detection = SpikeDetection(CustomSettingsSDA)
    if data_dict:
        process_dictionary(data_dict)

        cleaned_signals = data_dict.get("cleaned_signals")
        if cleaned_signals:
            processed_signals = []
            for signal in cleaned_signals:
                if isinstance(signal, np.ndarray):
                    processed_signal = spike_detection.sda_neo(signal)
                    processed_signals.append(processed_signal)
            if "processed_signals" not in data_dict:
                data_dict["processed_signals"] = processed_signals
            else:
                print("'processed_signals' bereits vorhanden, bestehende Daten werden nicht überschrieben.")
        else:
            print("Key 'cleaned_signals' fehlt!")

