import os
import numpy as np
import scipy.io
import numpy
from scipy.io import savemat
from scipy.io import loadmat


matlab_file_path = os.path.join("..", "..", "data", "2024-02-05_Dataset-KlaesNeuralDecoding (2).mat")

loaded_data = loadmat(matlab_file_path)

waveform_data = {}

for exp_key in loaded_data:
    if exp_key.startswith("exp_"):
        exp_data = loaded_data[exp_key][0, 0]

        experiment_waveforms = []

        for trial_key in exp_data.dtype.names:
            if trial_key.startswith("trial_"):
                trial_data = exp_data[trial_key][0, 0]

                waveforms = trial_data["waveforms"].flatten()
                experiment_waveforms.append(waveforms)

        waveform_data[exp_key] = experiment_waveforms

output_file_path = os.path.join(os.path.dirname(matlab_file_path), "_waveforms")
savemat(output_file_path + ".mat", waveform_data)
numpy.save(output_file_path + ".npy", waveform_data)
