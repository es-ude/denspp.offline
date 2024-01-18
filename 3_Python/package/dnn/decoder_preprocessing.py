import numpy as np
from package.metric import calculate_snr
from package.data.process_noise import frame_noise
import scipy.io as sio

# Dateipfade zu den MATLAB-Dateien im Ordner "Projekt"
behavior_file_path = 'Desktop/Forschungsprojekt/spaike_denssp/Projekt/behaviour_data2.mat'
nev_file_path = 'Desktop/Forschungsprojekt/spaike_denssp/Projekt/nev_data2.mat'

# Laden der Verhaltensdaten aus der MATLAB-Datei 'behaviour_data2.mat'
behavior_mat_contents = sio.loadmat(behavior_file_path)
behavior_data = behavior_mat_contents[
    'your_behavior_variable_name']  # Ersetzen Sie 'your_behavior_variable_name' durch den tatsächlichen Namen Ihrer Variable

# Laden der Spike-Daten aus der MATLAB-Datei 'nev_data2.mat'
nev_mat_contents = sio.loadmat(nev_file_path)
spike_data = nev_mat_contents[
    'your_spike_variable_name']  # Ersetzen Sie 'your_spike_variable_name' durch den tatsächlichen Namen Ihrer Variable

# Annahme: Abtastrate in Hertz
sampling_rate = 1000  # Beispielwert, 1000 Hz

# Berechnung der Anzahl der Samples für 100 ms
samples_per_window = int(sampling_rate * 0.1)  # 0.1 Sekunden entsprechen 100 ms

# Initialisierung der Start- und Endindizes
start_index = 0
end_index = samples_per_window

# Liste zum Speichern der ausgewählten Zeitfenster
selected_windows = []

# Schleife über alle Zeitfenster für Spike-Daten
while end_index <= len(spike_data[0]):
    # Anwenden der Funktion auf das aktuelle Zeitfenster
    selected_frames = change_frame_size(spike_data, [start_index, end_index])

    # Speichern der ausgewählten Frames
    selected_windows.append(selected_frames)

    # Verschiebung zu nächsten Zeitfenster
    start_index = end_index
    end_index = start_index + samples_per_window

# Liste zum Speichern der ausgewählten Zeitfenster für Verhaltensdaten
behavior_selected_windows = []

# Schleife über alle Zeitfenster für Verhaltensdaten
start_index = 0
end_index = samples_per_window

while end_index <= len(behavior_data[0]):
    # Anwenden der Funktion auf das aktuelle Zeitfenster
    selected_frames = change_frame_size(behavior_data, [start_index, end_index])

    # Speichern der ausgewählten Frames
    behavior_selected_windows.append(selected_frames)

    # Verschiebung zu nächsten Zeitfenster
    start_index = end_index
    end_index = start_index + samples_per_window
