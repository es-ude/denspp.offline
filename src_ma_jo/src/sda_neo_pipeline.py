from pathlib import Path
import numpy as np
from src_neuro.sda.sda_pipeline import Pipeline_Digital


def get_path(data_subdir="data", output_subdir="output", data_file_name="A1R1a_elec_stim_50biphasic_400us0001.npy"):
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
    artifact_indices = []
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
                    artifact_indices.append(indices)
                else:
                    print("  - Keine 'artifacts' gefunden.")
            else:
                print("  - Keine 'timestamps' gefunden.")
    else:
        print("Keine Details im Dictionary gefunden.")
    return artifact_indices

def compare_indices_in_loop(spike_index_array, artifact_index_array):
    # Überprüfe, ob beide Arrays die gleiche Länge haben
    if len(spike_index_array) != len(artifact_index_array):
        raise ValueError("Die Arrays haben unterschiedliche Längen und können nicht verglichen werden.")

    # Schleife durch beide Arrays
    for i in range(len(spike_index_array)):
        spike_indices = spike_index_array[i]
        artifact_indices = artifact_index_array[i]

        # Finde Überschneidungen der Indizes
        common_indices = np.intersect1d(spike_indices, artifact_indices)

        if common_indices.size > 0:
            print(f"Index {i}: Es gibt {len(common_indices)} Überschneidungen: {common_indices}")
        else:
            print(f"Index {i}: Keine Überschneidungen gefunden.")

    print("Vergleich der Arrays abgeschlossen.")



def get_spike_indices(cleaned_signals, fs=100):
    spike_indices = []
    pipeline = Pipeline_Digital(fs)
    pipeline.define_sda(mode_sda=1, mode_thr=2)  # Beispiel: NEO + CONST
    for idx, signal in enumerate(cleaned_signals):
        if isinstance(signal, np.ndarray):
            print(f"\nVerarbeite Signal {idx+1}/{len(cleaned_signals)}")
            pipeline.run_preprocess(signal, do_smooth=True,do_get_frames=True)
            print(f"  - Verwendete Methoden: {pipeline.used_methods}")
            print(f"  - SDA Output Shape: {pipeline.x_pos.shape}")
            print(f"  - Threshold Shape: {pipeline.signals.x_thr.shape}")
            print(pipeline.x_pos)
            spike_indices.append(pipeline.x_pos)
    return spike_indices

if __name__ == "__main__":
    file_path, output_dir = get_path()
    data_dict = load_file_as_dict(file_path)

    if data_dict:
        artifact_indices = process_dictionary(data_dict)

        cleaned_signals = data_dict.get("cleaned_signals")
        if cleaned_signals:
            if isinstance(cleaned_signals, list) and all(isinstance(s, np.ndarray) for s in cleaned_signals):
                spikes = get_spike_indices(cleaned_signals, fs=25e3)
            else:
                raise Exception("Fehler: 'cleaned_signals' ist kein gültiger Array-Container.")
        else:
            raise Exception("Key 'cleaned_signals' fehlt!")

        compare_indices_in_loop(spikes, artifact_indices)
