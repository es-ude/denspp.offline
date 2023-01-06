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

# TODO: Mehrkanal-Einlesung und Auswertung einfügen
def call_data(path2data: str, data_type: int, data_set: int, desired_fs: int, t_range: np.array, ch_sel: int, plot: bool) -> [np.array, np.array]:
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
        (data_in, label_in) = load_03_SimDaten_Quiroga2020(path2data, data_set)
        PrintOut = "Quiroga2020"
    elif data_type == 4:
        (data_in, label_in) = [None]
        PrintOut = "Freiburg2012"
    elif data_type == 5:
        (data_in, label_in) = load_05_Data_Klaes(path2data)
        PrintOut = "KlaesUSA"

    # --- Processing of raw data
    neuron.type = data_in["type"]
    neuron.channel = data_in["channel"]
    neuron.orig_fs = data_in["sampling_rate"]
    neuron.gain = data_in["gain"]
    neuron.rawdata = data_in["raw_data"][ch_sel] / neuron.gain

    # --- Processing of labeling informations
    labeling.exist = 1
    labeling.orig_spike_xpos = label_in["spike_times"]
    labeling.orig_cluster_id = label_in["spike_cluster"]
    labeling.orig_cluster_no = np.unique(labeling.orig_cluster_id).size

    # --- Cutting specific time range out of raw data
    if t_range.size == 2:
        idx0 = int(t_range[0] * neuron.orig_fs)
        idx1 = int(t_range[1] * neuron.orig_fs)
        data0 = neuron.rawdata[idx0:idx1]
    else:
        data0 = neuron.rawdata[:]

    # --- Resampling the input
    if desired_fs != neuron.orig_fs and desired_fs != 0:
        u_safe = 5e-6
        u_chck = np.mean(data0[0:10])
        if np.abs((u_chck < u_safe) - 1) == 1:
            du = u_chck
        else:
            du = 0

        (p, q) = get_resample_ratio(neuron.orig_fs, desired_fs)
        data1 = du + resample_poly(data0 - du, p, q)
        scaling = p / q
        neuron.data = (data1 + du)
        neuron.fs = desired_fs
    else:
        neuron.data = data0
        scaling = 1
        neuron.fs = neuron.orig_fs

    # --- "Resampling" the labeled informations
    if labeling.exist:
        if t_range.size == 2:
            # Find values from x-positions
            idx2 = np.argwhere(labeling.orig_spike_xpos >= idx0)
            idx2 = 1+int(idx2[0])
            idx3 = np.argwhere(labeling.orig_spike_xpos <= idx1)
            idx3 = int(idx3[-1])
        else:
            idx2 = 0
            idx3 = -1

        # Applying
        labeling.cluster_id = labeling.orig_cluster_id[idx2:idx3]
        labeling.cluster_no = np.unique(labeling.orig_cluster_id).size
        labeling.spike_xpos = scaling * labeling.orig_spike_xpos[idx2:idx3]
        TextLabel = "includes labels (noSpikes: " + str(np.size(labeling.spike_xpos)) + " - noCluster: " + str(labeling.cluster_no) + ")"
    else:
        TextLabel = "excludes labels"

    # ---- Output of meta informations
    length_orig = neuron.rawdata.size
    length_new = neuron.data.size / scaling
    if plot:
        print("... using data set of:", PrintOut)
        print("... original sampling rate of", int(1e-3 * neuron.orig_fs), "kHz", "(resampling to", int(1e-3*neuron.fs), "kHz)")
        print("... using", round(length_new / length_orig *100 , 2), "% of the data (time length of", round(neuron.rawdata.size / neuron.orig_fs, 2), "s)")
        print("... data includes", neuron.channel, "number of electrode (" + neuron.type + ") and", TextLabel)

    return (neuron, labeling)


def get_resample_ratio(fin: int, fout: int) -> Tuple[int, int]:
    # average sampling interval
    calced_fs = fout / fin
    (p, q) = Fraction(calced_fs).limit_denominator(100).as_integer_ratio()

    return (p, q)


def load_01_SimDaten_Martinez2009(
        path2data: str, indices: list = np.arange(1, 5)
) -> dict:
    folder = "01_SimDaten_Martinez2009"
    file_data = "simulation_" + str(indices) + ".mat"
    path2file = os.path.join(path2data, folder, file_data)
    data = dict()
    label = dict()

    print(["... using data point:", path2file])

    loaded_data = loadmat(path2file)
    data["type"] = "Synthetic"
    data["channel"] = int(loaded_data["chan"][0])
    data["gain"] = 10 ** (0 / 20)
    data["sampling_rate"] = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
    data["raw_data"] = 0.5e-6 * loaded_data["data"]

    label["spike_times"] = loaded_data["spike_times"][0][0][0]
    label["spike_cluster"] = loaded_data["spike_class"][0][0][0]

    return (data, label)


def load_02_SimDaten_Pedreira2012(
    path2data: str, indices: list = np.arange(1, 16)
) -> dict:
    folder = "02_SimDaten_Pedreira2012"
    folder_content = os.listdir(os.path.join(path2data, folder))

    file_data = folder_content[indices + 1]
    prep_index = file_data.split("_", 1)[1]
    num_index = int(prep_index[0:2])
    file_ground = "ground_truth.mat"
    path2file = os.path.join(path2data, folder, file_data)
    path2ground = os.path.join(path2data, folder, file_ground)
    data = dict()
    label = dict()

    print(["... using data point:", path2file, num_index])

    loaded_data = loadmat(path2file)
    data["type"] = "Synthetic"
    data["channel"] = int(1)
    data["gain"] = 10 ** (0 / 20)
    data["sampling_rate"] = int(24000)
    data["raw_data"] = 25e-6 * loaded_data["data"]

    ground_truth = loadmat(path2ground)
    label["spike_times"] = ground_truth["spike_first_sample"][0][num_index - 1][0]
    label["spike_cluster"] = ground_truth["spike_classes"][0][num_index - 1][0]

    return (data, label)


def load_03_SimDaten_Quiroga2020(
        path2data: str, indices: list = np.arange(1, 22)
) -> dict:
    folder = "03_SimDaten_Quiroga2020"
    path2folder = os.path.join(path2data, folder)
    files = os.listdir(path2folder)
    files.sort()
    file = files[indices]
    path2file = os.path.join(path2folder, file)

    data = dict()
    label = dict()

    print(["... using data point:", path2file])

    loaded_data = loadmat(path2file)
    data["type"] = "Synthetic"
    data["channel"] = int(1)
    data["gain"] = 10 ** (0 / 20)
    data["sampling_rate"] = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
    data["raw_data"] = 100e-6 * loaded_data["data"]

    label["spike_times"] = loaded_data["spike_times"][0][0][0]
    label["spike_cluster"] = loaded_data["spike_class"][0][0][0]-1

    return (data, label)


# TODO: Andere Quellen noch anpassen
def load_05_Data_Klaes(
    path2data: str
) -> None:
    folder = "05_Daten_Klaes"
    path2folder = os.path.join(path2data, folder)
    sessions = list(set(os.listdir(path2folder)).difference(set([".DS_Store", ".", ".."])))
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
