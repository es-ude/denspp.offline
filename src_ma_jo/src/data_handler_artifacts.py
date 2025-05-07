import os
import numpy as np
from scipy.io import loadmat


def load_data(path, filename):
    """Lädt eine .mat-Datei und gibt deren Inhalt zurück."""
    full_path = os.path.join(path, filename)
    return loadmat(full_path)


def extract_arrays(data, key):
    """
    Extrahiert eine bestimmte Anzahl von Arrays aus den geladenen Daten.
    Args:
        data: Die geladene Datenstruktur.
        key: Schlüssel, der die Arrays in den Daten referenziert.
        num_arrays: Anzahl der Arrays, die extrahiert werden sollen.

    Returns:
        Liste der extrahierten Arrays.
    """
    num_arrays = len(data[key][2])
    return [np.array([arr[i] for arr in data[key]]) for i in range(num_arrays)]
