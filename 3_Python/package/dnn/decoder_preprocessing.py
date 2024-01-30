import scipy.io as sio
from scipy.io import loadmat
import os
import numpy as np
from package.metric import calculate_snr
from package.data_call.process_noise import frame_noise
from package.data_call.data_call_cellbib import CellSelector
from sklearn.preprocessing import StandardScaler #Standardizes features by removing the mean and scaling to unit variance.
from sklearn.preprocessing import MinMaxScaler #Scales features to a specified range, usually between 0 and 1.
from sklearn.impute import SimpleImputer #Handles missing values by imputing them with mean, median, or most frequent values.

import os
import scipy.io
import numpy as np

# Erstellen Sie das Hauptverzeichnis
os.makedirs('Test/Try/Elektrode1', exist_ok=True)

# Speichern Sie die timestamps-Daten
timestamps = np.array([100, 200, 300, 400, 500])
scipy.io.savemat('Test/Try/Elektrode1/timestamps.mat', {'timestamps': timestamps})

# Speichern Sie die spikes-Daten
spikes = np.array([(0, 0), (30, 1), (50, 1), (70, 0), (100, 0), (140, 1), (170, 1),
                   (200, 1), (220, 1), (250, 0), (270, 1), (300, 1), (310, 0),
                   (340, 1), (360, 1), (380, 1), (400, 1), (430, 0), (480, 1), (500, 1)])
scipy.io.savemat('Test/Try/Elektrode1/spikes.mat', {'spikes': spikes})

# Erstellen Sie die zusätzlichen Verzeichnisse
os.makedirs('Test/Try/Elektrode2', exist_ok=True)
os.makedirs('Test/Try/Elektrode3', exist_ok=True)

# Speichern Sie die Spikes-Daten für Elektrode2
spikes2 = np.array([(0, 0), (20, 1), (34, 1), (48, 1), (56, 1), (67, 0),
                    (89, 0), (103, 1), (123, 1), (134, 1), (156, 0), (168, 0),
                    (197, 1), (205, 1)])
scipy.io.savemat('Test/Try/Elektrode2/spikes.mat', {'spikes': spikes2})

# Speichern Sie die Spikes-Daten für Elektrode3
spikes3 = np.array([(0, 0), (49, 1), (78, 0), (309, 1), (805, 1)])
scipy.io.savemat('Test/Try/Elektrode3/spikes.mat', {'spikes': spikes3})


def calculate_output(spikes):
    output = [[], [], []]
    current_section = 100
    ones_count = 0
    zeros_count = 0

    for pair in spikes:
        while pair[0] >= current_section:
            output[0].append(current_section)
            output[1].append(ones_count)
            output[2].append(zeros_count)
            current_section += 100
            ones_count = 0
            zeros_count = 0
        if pair[1] == 1:
            ones_count += 1
        else:
            zeros_count += 1

    while current_section <= spikes[-1, 0] + 100:
        output[0].append(current_section)
        output[1].append(ones_count)
        output[2].append(zeros_count)
        current_section += 100

    return output


spikes1 = np.array(
    [(0, 0), (1, 1), (20, 1), (34, 1), (48, 1), (56, 1), (67, 0), (89, 0), (103, 1), (123, 1), (134, 1), (156, 0),
     (168, 0), (197, 1), (205, 1), (300, 0), (400, 0), (500, 0), (600, 1)])
spikes2 = np.array(
    [(0, 0), (20, 1), (34, 1), (48, 1), (56, 1), (67, 0), (89, 0), (103, 1), (123, 1), (134, 1), (156, 0), (168, 0),
     (197, 1), (205, 1)])
spikes3 = np.array([(0, 0), (49, 1), (78, 0), (309, 1), (805, 1)])

output1 = calculate_output(spikes1)
output2 = calculate_output(spikes2)
output3 = calculate_output(spikes3)

final_output = {
    'Elektrode1': output1,
    'Elektrode2': output2,
    'Elektrode3': output3,
}

for key, value in final_output.items():
    print(f"{key}: \nAbschnitte: {value[0]} \nAnzahl der 1en: {value[1]} \nAnzahl der 0en: {value[2]} \n")


