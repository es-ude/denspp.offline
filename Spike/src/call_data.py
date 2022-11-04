import numpy as np
import os
from scipy.io import loadmat
import mat73

# ----- Read Settings -----
class Labeling:
    exist = 0
    orig_cluster_id = None
    orig_cluster_no = None
    orig_spike_xpos = None
    orig_spike_no = None

class NeuroInput:
    type = None
    gain = None
    origFs = None
    channel = None
    fs = None
    time = None
    rawdata = None
    data = None

def callData(Path2Data: str, DataType: int, DataSet: int, desired_fs: int, t_range: np.array):
    labeling = Labeling()
    neuron = NeuroInput()

    # ----- Read data input -----
    match DataType:
        case 1:
            (dataIn, labelIn) = load_01_SimDaten_Martinez2009(Path2Data, DataSet)
        case 2:
            (dataIn, labelIn) = load_02_SimDaten_Pedreira2012(Path2Data, DataSet)
        case 3:
            print("C")
        case 4:
            print("D")

    # --- Processing of raw data
    neuron.type = dataIn["type"]
    neuron.channel = dataIn["channel"]
    neuron.origFs = dataIn["sampling_rate"]
    neuron.gain_pre = dataIn["gain"]
    neuron.rawdata = dataIn["raw_data"] / neuron.gain_pre
    t0 = np.arange(0, neuron.rawdata.size - 1, 1).reshape((1, neuron.rawdata.size - 1)) / neuron.origFs

    # --- Processing of labeling informations
    spike_class = labelIn["spike_cluster"]
    spike_times = labelIn["spike_times"]

    labeling.exist = 1
    labeling.orig_spike_xpos = spike_times
    labeling.orig_spike_no = spike_times.size
    labeling.orig_cluster_id = spike_class
    labeling.orig_cluster_no = np.unique(spike_class).size

    # --- Cutting the values out of array
    if t_range.size == 2:
        idx0 = np.argwhere(t_range[0, 0] <= t0)
        idx0 = int(idx0[0, 1])
        idx1 = np.argwhere(t_range[0, 1] <= t0)
        idx1 = int(idx1[0, 1])

        neuron.time = np.linspace(0, idx1-idx0-1, idx1-idx0)/neuron.origFs
        neuron.data = neuron.rawdata[idx0:idx1]
    else:
        neuron.time = t0
        neuron.data = neuron.data

    #TODO: Neuabtastung einfügen

    return (neuron, labeling)

def load_01_SimDaten_Martinez2009(Path2Data: str, indices: list = [1, 2, 3, 4, 5]):
    folder = '01_SimDaten_Martinez2009'
    file = 'simulation_' + str(indices) + '.mat'
    Path2File = os.path.join(Path2Data, folder, file)
    data = dict()
    label = dict()

    loaded_data = loadmat(Path2File)

    data['type'] = "Synthetic"
    data['channel'] = int(loaded_data['chan'][0])
    data['gain'] = 10 ** (0 / 20)
    data['sampling_rate'] = int(1/loaded_data['samplingInterval'][0][0]*1000)
    data['raw_data'] = 0.5e-6* loaded_data['data'][0]

    label['spike_times'] = loaded_data['spike_times'][0][0][0]
    label['spike_cluster'] = loaded_data['spike_class'][0][0][0]

    return (data, label)

def load_02_SimDaten_Pedreira2012(Path2Data: str, indices: list=[1,2,3,4,5,6,7,8,9,10,90,91,92,93,94,95]) -> dict:
    folder = '02_SimDaten_Pedreira2012'
    fileData = 'simulation_' + str(indices) + '.mat'
    fileGround = 'ground_truth.mat'
    Path2File = os.path.join(Path2Data, folder, fileData)
    Path2Ground = os.path.join(Path2Data, folder, fileGround)
    data = dict()
    label = dict()

    loaded_data = loadmat(Path2File)
    data['type'] = "Synthetic"
    data['channel'] = int(1)
    data['gain'] = 10 ** (0 / 20)
    data['sampling_rate'] = float(24000)
    data['raw_data'] = 25e-6* loaded_data['data'][0]

    ground_truth = loadmat(Path2Ground)
    label['spike_times'] = ground_truth['spike_first_sample'][0][indices-1][0]
    label['spike_cluster'] = ground_truth['spike_classes'][0][indices-1][0]

    return (data, label)

#TODO: Andere Quellen noch anpassen
def load_03_SimDaten_Quiroga2020(Path2Data: str, indices: list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]):
    folder = ['03_SimDaten_Quiroga2020']
    files = os.listdir(folder)
    files.sort()
    data = dict()
    label = dict()

    for i in indices:
        file = files[i-1]
        path = os.path.join(folder, file)
        loaded_data = loadmat(path)
        data = dict()
        print(loaded_data.keys())
        print(loaded_data['spikes'])
        data['sampling_rate'] = float(1 / loaded_data['samplingInterval'][0][0] * 1000)
        data['raw_data'] = loaded_data['data'][0]
        data['spike_times'] = loaded_data['spike_times'][0][0][0]
        data['spike_cluster'] = loaded_data['spike_class'][0][0][0]
        data_list.append(data)
    return data_list

def load_05_Data_Klaes() -> None:
    folder = ['2_Data', '05_Daten_Klaes']
    sessions = list(set(os.listdir(folder)).difference(set(['.DS_Store', '.', '..'])))
    expected_subfolders = ['NSP1', 'NSP2']
    for session in sessions:
        for subfolder in expected_subfolders:
            path = os.sep.join([folder, session, subfolder])
            files = os.listdir(path)
            recordings = [x for x in files if '.mat' in x]
            for record in recordings:
                if '102124-NSP1' in record:
                    path_data = os.sep.join([path, record])
                    path_metadata = path_data.replace('.mat', '.ccf')
                    data = mat73.loadmat(path_data)