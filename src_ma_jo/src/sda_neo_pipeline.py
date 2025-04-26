from pathlib import Path
import numpy as np
from src_neuro.sda.sda_pipeline import Pipeline_Digital
from denspp.offline.digital.sda import SpikeDetection, SettingsSDA
from denspp.offline.data_process.transformation import window_method

def get_path(data_subdir="data", output_subdir="output", data_file_name="A1R1a_elec_stim_50biphasic_400us0001.npy"):
    project_base_dir = Path(__file__).resolve().parent.parent
    data_dir_path = project_base_dir / data_subdir
    output_dir_path = project_base_dir / output_subdir

    data_file_path = data_dir_path / data_file_name

    if not data_file_path.exists():
        print(f"Die Datei existiert NICHT: {data_file_path}")
    if not output_dir_path.exists():
        print(f"Das Ausgabeverzeichnis existiert NICHT: {output_dir_path}")

    return data_file_path, output_dir_path

def load_file_as_dict(input_file_path):
    if not Path(input_file_path).exists():
        print(f"Fehler: Datei wurde nicht gefunden: {input_file_path}")
        return None

    try:
        data = np.load(input_file_path, allow_pickle=True).item()
        print(f".npy-Datei erfolgreich als Dictionary geladen: {input_file_path}")
        return data
    except Exception as e:
        print(f"Fehler beim Laden der Datei: {file_path}. Details: {e}")
        return None

def process_dictionary(local_data_dict):
    artifact_indices = []
    if not isinstance(local_data_dict, dict):
        print("Fehler: Geladene Daten sind kein Dictionary!")
        return

    print(f"Keys im geladenen Dictionary: {list(local_data_dict.keys())}")

    details = local_data_dict.get("details")
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

def compare_indices_in_loop(spike_indices_list, artifact_indices_list):
    if len(spike_indices_list) != len(artifact_indices_list):
        raise ValueError("Die Arrays haben unterschiedliche Längen und können nicht verglichen werden.")

    for i in range(len(spike_indices_list)):
        spikes = spike_indices_list[i]
        artifacts = artifact_indices_list[i]

        # Finde Überschneidungen der Indizes
        common_indices = np.intersect1d(spikes, artifacts)

        if common_indices.size > 0:
            print(f"Index {i}: Es gibt {len(common_indices)} Überschneidungen: {common_indices}")
        else:
            print(f"Index {i}: Keine Überschneidungen gefunden.")

    print("Vergleich der Arrays abgeschlossen.")

def get_spike_indices_existing_pipeline(cleaned_signals, fs=100):
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

    # --------- Pre-Processing of SDA -------------
def time_delay(uin, fs):
    """Applying a time delay on the input signal"""
    t_dly = 0.3e-3  #Wert aus Settings
    set_delay = round(t_dly * settings_sda.fs)
    mat = np.zeros(shape=(set_delay,), dtype=float)
    uout = np.concatenate((mat, uin[0:uin.size - set_delay]), axis=None)
    return uout

def sda_neo(xin):
    """Applying Non-Linear Energy Operator (NEO, same like Teager-Kaiser-Operator) with dx_sda = 1 or kNEO with dx_sda > 1"""
    ksda0 = settings_sda.dx_sda[0]
    x_neo0 = np.floor(xin[ksda0:-ksda0] ** 2 - xin[:-2 * ksda0] * xin[2 * ksda0:])
    x_neo = np.concatenate([x_neo0[:ksda0, ], x_neo0, x_neo0[-ksda0:, ]], axis=None)
    return x_neo

def sda_smooth(xin: np.ndarray, window_method='Hamming') -> np.ndarray:
    """Smoothing the input with defined window ['Hamming', 'Gaussian', 'Flat', 'Bartlett', 'Blackman']"""
    return spike_detector.smoothing_1d(xin, 4 * 1 + 1, window_method)

def thres_rms(xin: np.ndarray) -> np.ndarray:
    """Applying the root-mean-squre (RMS) on neural input"""
    C = settings_sda.thr_gain
    M = settings_sda.window_size
    return C * np.sqrt(np.convolve(xin ** 2, np.ones(M) / M, mode='same'))


if __name__ == "__main__":
    settings_sda = SettingsSDA(
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
    file_path, output_dir = get_path()
    loaded_data_dict = load_file_as_dict(file_path)
    spike_detector = SpikeDetection(settings_sda)

    if loaded_data_dict:
        artifact_indices = process_dictionary(loaded_data_dict)

        cleaned_signals = loaded_data_dict.get("cleaned_signals")
        if cleaned_signals:
            if isinstance(cleaned_signals, list) and all(isinstance(s, np.ndarray) for s in cleaned_signals):
                #spike_indices_list = get_spike_indices_existing_pipeline(cleaned_signals, fs=25e3)
                for signal in cleaned_signals:
                    x_dly = time_delay(signal, settings_sda.fs)
                    x_sda = sda_neo(x_dly)
                    smooth_data = sda_smooth(x_sda, "Hamming")
                    threhold_data = thres_rms(smooth_data)
            else:
                raise Exception("Fehler: 'cleaned_signals' ist kein gültiger Array-Container.")
        else:
            raise Exception("Key 'cleaned_signals' fehlt!")

        compare_indices_in_loop(spike_indices_list, artifact_indices)
