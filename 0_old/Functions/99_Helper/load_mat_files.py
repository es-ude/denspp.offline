import os, sys
from scipy.io import loadmat
import mat73
import numpy as np

def load_mat_file(path: str) -> dict:
    data = loadmat(path)
    # print(data['data'])
    # print(data['data'].shape)
    # print(type(data['data']))
    # print()
    # print(data['spike_times'])
    # print(data['spike_times'][0][0])
    # print(data['spike_times'][0][0].shape)
    # print(type(data['spike_times'][0][0]))
    # print()
    # print(data['spike_class'])
    # print(data['spike_class'][0][0])
    # print(np.column_stack(np.unique(data['spike_class'][0][0], return_counts=True)))
    # print(type(data['spike_class']))
    return data


def load_01_SimDaten_Martinez2009(indices: list = [1, 2, 3, 4, 5]) -> dict:
    folder = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2] + ['2_Data', '01_SimDaten_Martinez2009'])
    files = ['simulation_' + str(i) + '.mat' for i in indices]
    data_list = list()
    for file in files:
        path = os.path.join(folder, file)
        loaded_data = load_mat_file(path)
        data = dict()
        data['sampling_rate'] = float(1/loaded_data['samplingInterval'][0][0]*1000)
        data['raw_data'] = loaded_data['data'][0]
        data['spike_times'] = loaded_data['spike_times'][0][0][0]
        data['spike_cluster'] = loaded_data['spike_class'][0][0][0]
        data_list.append(data)
    return data_list

def load_02_SimDaten_Pedreira2012(indices: list=[1,2,3,4,5,6,7,8,9,10,90,91,92,93,94,95]) -> dict:
    folder = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2] + ['2_Data', '02_SimDaten_Pedreira2012'])
    ground_truth_path = os.path.join(folder, 'ground_truth.mat')
    ground_truth = load_mat_file(ground_truth_path)
    data_list = list()
    for i in indices:
        file = 'simulation_' + str(i) + '.mat'
        path = os.path.join(folder, file)
        loaded_data = load_mat_file(path)

        data = dict()
        data['sampling_rate'] = float(24000)
        data['raw_data'] = loaded_data['data'][0]
        data['spike_times'] = ground_truth['spike_first_sample'][0][i - 1][0]
        data['spike_cluster'] = ground_truth['spike_classes'][0][i - 1][0]
        data_list.append(data)
    return data_list

def load_03_SimDaten_Quiroga2020(indices: list=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]):
    folder = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2] + ['2_Data', '03_SimDaten_Quiroga2020'])
    files = os.listdir(folder)
    files.sort()
    data_list = list()
    for i in indices:
        file = files[i-1]
        path = os.path.join(folder, file)
        loaded_data = load_mat_file(path)
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
    folder = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2] + ['2_Data', '05_Daten_Klaes'])
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

