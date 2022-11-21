import numpy as np
import os
from typing import Tuple
from fractions import Fraction

from scipy.io import loadmat
from scipy.signal import resample_poly
import mat73

# ----- Read Settings -----
class Labeling:
    exist = 0
    # From original data (comparing with rawdata)
    orig_cluster_id = None
    orig_cluster_no = None
    orig_spike_xpos = None
    # Apply for output data
    cluster_id = None
    cluster_no = None
    spike_xpos = None

class NeuroInput:
    type = None
    gain = None
    orig_fs = None
    channel = None
    fs = None
    rawdata = None
    data = None

# TODO: Anpassungen auf mehrere Elektroden-Känale (bisher nur ein Kanal)
def call_data(path2data: str, data_type: int, data_set: int, desired_fs: int, t_range: np.array) -> [np.array, np.array]:
    labeling = Labeling()
    neuron = NeuroInput()

    # ----- Read data input -----
    if data_type == 1:
        (data_in, label_in) = load_01_SimDaten_Martinez2009(path2data, data_set)
        PrintOut = "Martinez2009"
    elif data_type == 2:
        (data_in, label_in) = load_02_SimDaten_Pedreira2012(path2data, data_set)
        PrintOut = "Pedreira2012"
    elif data_type == 3:
        (data_in, label_in) = [None]
        PrintOut = None
    elif data_type == 4:
        (data_in, label_in) = [None]
        PrintOut = None

    # --- Processing of raw data
    neuron.type = data_in["type"]
    neuron.channel = data_in["channel"]
    neuron.orig_fs = data_in["sampling_rate"]
    neuron.gain_pre = data_in["gain"]
    neuron.rawdata = data_in["raw_data"] / neuron.gain_pre

    # --- Processing of labeling informations
    spike_class = label_in["spike_cluster"]
    spike_times = label_in["spike_times"]

    labeling.exist = 1
    labeling.orig_spike_xpos = spike_times
    labeling.orig_cluster_id = spike_class
    labeling.orig_cluster_no = np.unique(spike_class).size

    # --- Cutting specific time range out of raw data
    t0 = np.arange(0, neuron.rawdata.size, 1).reshape((1, -1)) / neuron.orig_fs
    if t_range.size == 2:
        idx0 = np.argwhere(t_range[0] <= t0)
        idx0 = int(idx0[0, 1])
        idx1 = np.argwhere(t_range[1] <= t0)
        idx1 = int(idx1[0, 1])

        t0 = t0[[0], idx0:idx1]
        data0 = neuron.rawdata[0, idx0:idx1]
    else:
        t0 = t0
        data0 = neuron.rawdata

    # --- Resampling the input
    if desired_fs != neuron.orig_fs and desired_fs != 0:
        u_safe = 5e-6
        if np.abs(np.sum((np.mean(data0[0]) - np.array([-1, 1]) * u_safe) < 0) - 1) == 1:
            du = np.mean(data0[0])
        else:
            du = 0

        p, q = get_resample_ratio(t0, desired_fs)
        data1 = resample_poly(data0 - du, p, q)
        neuron.data = (data1 + du).reshape((1, -1))
        neuron.fs = desired_fs
    else:
        neuron.data = data0
        neuron.fs = neuron.orig_fs

    # --- "Resampling" the labeled informations
    if labeling.exist:
        TextLabel = "includes labels"
        # Find values from x-positions
        idx2 = np.argwhere(labeling.orig_spike_xpos <= idx0)
        idx2 = 1+int(idx2[-1])
        idx3 = np.argwhere(labeling.orig_spike_xpos <= idx1)
        idx3 = int(idx3[-1])

        # Applying
        labeling.cluster_id = spike_class[idx2:idx3]
        labeling.cluster_no = np.unique(labeling.orig_cluster_id).size
        labeling.spike_xpos = spike_times[idx2:idx3]
    else:
        TextLabel = "excludes labels"

    # ---- Output of meta informations
    length_orig = neuron.rawdata.size
    length_new = neuron.data.size* neuron.orig_fs/neuron.fs

    print("... original sampling rate of", int(1e-3 * neuron.orig_fs), "kHz", "(resampling to", int(1e-3*neuron.fs), "kHz)")
    print("... using", round(length_new / length_orig *100 , 2), "% of the data (time length of", round(neuron.rawdata.size/neuron.orig_fs, 2), "s)")
    print("... data includes", neuron.channel, "number of electrode (" + neuron.type + ") and", TextLabel)

    return (neuron, labeling)

def get_resample_ratio(t, fs) -> Tuple[int, int]:
    # average sampling interval
    calced_fs = ((t[0, -1] - t[0, 0]) / (t.size)) * fs
    # matlab 'rat' function
    p, q = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()

    return p, q

def load_01_SimDaten_Martinez2009(path2data: str, indices: list = [1, 2, 3, 4, 5]):
    folder = "01_SimDaten_Martinez2009"
    file = "simulation_" + str(indices) + ".mat"
    path2file = os.path.join(path2data, folder, file)
    data = dict()
    label = dict()

    loaded_data = loadmat(path2file)

    data["type"] = "Synthetic"
    data["channel"] = int(loaded_data["chan"][0])
    data["gain"] = 10 ** (0 / 20)
    data["sampling_rate"] = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
    data["raw_data"] = 0.5e-6 * loaded_data["data"] / data["gain"]

    label["spike_times"] = loaded_data["spike_times"][0][0][0]
    label["spike_cluster"] = loaded_data["spike_class"][0][0][0]

    return (data, label)


def load_02_SimDaten_Pedreira2012(
    path2data: str, indices: list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90, 91, 92, 93, 94, 95]
) -> dict:
    folder = "02_SimDaten_Pedreira2012"
    file_data = "simulation_" + str(indices) + ".mat"
    file_ground = "ground_truth.mat"
    path2file = os.path.join(path2data, folder, file_data)
    path2ground = os.path.join(path2data, folder, file_ground)
    data = dict()
    label = dict()

    loaded_data = loadmat(path2file)
    data["type"] = "Synthetic"
    data["channel"] = int(1)
    data["gain"] = 10 ** (0 / 20)
    data["sampling_rate"] = float(24000)
    data["raw_data"] = 25e-6 * loaded_data["data"][0]

    ground_truth = loadmat(path2ground)
    label["spike_times"] = ground_truth["spike_first_sample"][0][indices - 1][0]
    label["spike_cluster"] = ground_truth["spike_classes"][0][indices - 1][0]

    return (data, label)

# TODO: Andere Quellen noch anpassen
def load_03_SimDaten_Quiroga2020(path2data: str, indices: list = np.arange(1, 26)):
    data_list = None
    folder = ["03_SimDaten_Quiroga2020"]
    files = os.listdir(folder)
    files.sort()
    data = dict()
    label = dict()

    for i in indices:
        file = files[i - 1]
        path = os.path.join(folder, file)
        loaded_data = loadmat(path)
        data = dict()
        print(loaded_data.keys())
        print(loaded_data["spikes"])
        data["sampling_rate"] = float(1 / loaded_data["samplingInterval"][0][0] * 1000)
        data["raw_data"] = loaded_data["data"][0]
        data["spike_times"] = loaded_data["spike_times"][0][0][0]
        data["spike_cluster"] = loaded_data["spike_class"][0][0][0]
        data_list.append(data)
    return data_list


def load_05_Data_Klaes() -> None:
    folder = ["2_Data", "05_Daten_Klaes"]
    sessions = list(set(os.listdir(folder)).difference(set([".DS_Store", ".", ".."])))
    expected_subfolders = ["NSP1", "NSP2"]
    for session in sessions:
        for subfolder in expected_subfolders:
            path = os.sep.join([folder, session, subfolder])
            files = os.listdir(path)
            recordings = [x for x in files if ".mat" in x]
            for record in recordings:
                if "102124-NSP1" in record:
                    path_data = os.sep.join([path, record])
                    path_metadata = path_data.replace(".mat", ".ccf")
                    data = mat73.loadmat(path_data)
