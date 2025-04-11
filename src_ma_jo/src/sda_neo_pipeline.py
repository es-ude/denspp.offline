from pathlib import Path
import numpy as np


def get_path(data_subdir="data", output_subdir="output", data_file_name="A1R1a_elec_stim_50biphasic_400us0001.npy"):
    """
    Dynamisch generiert Pfade relativ zur Datei-Position des Skripts.

    :param data_subdir: Name des Datenunterordners
    :param output_subdir: Name des Ausgabeunterordners
    :param data_file_name: Name der Zieldatei
    :return: Tuple (Datenpfad, Ausgabeordner)
    """
    base_dir = Path(__file__).resolve().parent.parent  # Gehe 1 Ebene über das aktuelle Skript hinaus

    data_dir = base_dir / data_subdir
    output_dir = base_dir / output_subdir
    data_file = data_dir / data_file_name

    # Validierung der Existenz der Datei und Verzeichnisse
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
        # allow_pickle=True, um gespeicherte Python-Objekte (z.B. Dictionaries) zu laden
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

    # Ausgabe aller Keys im obersten Dictionary-Level
    print(f"Keys im geladenen Dictionary: {list(data_dict.keys())}")

    # Extra: Zugriff auf Daten in 'details'
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

    if data_dict:
        process_dictionary(data_dict)

        cleaned_signals = data_dict.get("cleaned_signals")
        if cleaned_signals:
            print(cleaned_signals[1])
        else:
            print("Key 'cleaned_signals' fehlt!")

