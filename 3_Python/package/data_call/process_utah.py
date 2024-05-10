import os
import glob
import numpy as np
from neo.io import BlackrockIO
from scipy.io import savemat
from scipy.io import loadmat as scipy_loadmat
from mat73 import loadmat as mat73_loadmat


class UtahDataHandler:
    def __init__(self, path: str, folder_name: str):
        self.__path2data = path
        self.__path2save = os.path.join(self.__path2data, folder_name)
        self.__path2orig = os.path.join(self.__path2save, 'MERCUR_Files')

        self.nsx_version = None
        self.noDataSets = 0
        self.noDataPoints = 0
        self.noElectrodes = 0
        self.folder_content = self.__get_folder_content()

    def __get_folder_content(self) -> list:
        folder_content = os.listdir(self.__path2orig)
        folder_content.sort()
        self.noDataSets = len(folder_content)

        return folder_content

    def search_data(self, sel_folder: int) -> dict:
        """Generate a dict overview about data sets"""
        # if sel_nsp == 0, then all NSP datasets will be loaded
        # Name of thresholding files ('*.nev', '*.mat') should be the same like from the experiment
        # '*.ccf'-Files will be ignored
        type_nsp = '*NSP*'
        type_behavior = '*behav*.mat'
        type_rawdata = '*.ns6'
        type_thres = '*.nev'

        path2data = self.__path2orig
        folder_content = os.listdir(path2data)

        # --- Searching for rawdata and behavior files
        files_nsp = glob.glob(os.path.join(path2data, folder_content[sel_folder], type_nsp))
        files_nsp.sort()
        files_behv = glob.glob(os.path.join(path2data, folder_content[sel_folder], type_behavior))
        files_behv.sort()

        # --- Sorting infos to right place
        path2nsx = []
        path2nev = []
        path2mat = []
        expe_name = []
        for idx, file in enumerate(files_nsp):
            check_ending = os.path.splitext(file)[-1][1:]
            if check_ending == type_rawdata[-3:]:
                path2nsx.append(file)
                expe_name.append(os.path.splitext(os.path.basename(file[0:-4]))[0])
            if check_ending == type_thres[-3:]:
                path2nev.append(file)
            if check_ending == 'mat':
                path2mat.append(file)

        path2behav = []
        for idx, file in enumerate(files_behv):
            check_ending = file[-3:]
            if check_ending == 'mat':
                path2behav.append(file)

        # --- Generate dictionary
        self.nsx_version = int(type_rawdata[-1])
        self.noDataPoints = len(path2nsx)
        dataset_path = dict()
        dataset_path['Name'] = expe_name
        dataset_path['NSX'] = path2nsx
        dataset_path['NoNSP'] = len(path2nsx)
        dataset_path['NEV'] = path2nev
        dataset_path['MAT'] = path2mat
        dataset_path['Behaviour'] = path2behav

        # --- Checking if something is available
        if len(path2behav) != 0:
            dataset_path['Exits_BEHV'] = True
        else:
            dataset_path['Exits_BEHV'] = False

        return dataset_path

    def process_rawdata(self, data_dict: dict, sel_file: int) -> dict:
        """Processing the rawdata"""
        neural_data = dict()

        path2file = data_dict['NSX'][sel_file]
        if not os.path.exists(path2file):
            neural_data["Exits"] = False
        else:
            neural_data["Exits"] = True
            data_in = BlackrockIO(path2file[:-4], nsx_to_load=self.nsx_version)

            # --- Reading rawdata (*.ns6)
            no_electrodes = data_in.header['signal_channels'].size
            self.noElectrodes = no_electrodes
            neural_data['NoElectrodes'] = no_electrodes
            neural_data['spike'] = data_in.nsx_datas[self.nsx_version][1]
            count_scale = 0
            count_fs = 0

            firstRun = True
            for idx in range(0, no_electrodes):
                meta_info = data_in.header['signal_channels']
                if firstRun:
                    meta_check = []
                    meta_check.append(meta_info[idx][2])
                    meta_check.append(meta_info[idx][4])
                    meta_check.append(meta_info[idx][5])
                else:
                    if meta_info[idx][2] == meta_check[0]:
                        count_fs += 1
                    if meta_info[idx][4] == meta_check[1] and meta_info[idx][5] == meta_check[2]:
                        count_scale += 1

                firstRun = False

            # print(count_fs / idx, count_scale / idx)
            if (count_fs / 95 >= 0.9):
                neural_data['SamplingRate'] = meta_info[0][2]
            if (count_scale / 95 >= 0.9):
                trennzeichen = ' '
                word_list = []
                word_list.append(meta_info[0][5].astype('str'))
                word_list.append(meta_info[0][4])
                neural_data['LSB'] = trennzeichen.join(word_list)

        return neural_data

    def process_label_nev(self, data_dict: dict, sel_file: int) -> dict:
        """Process '*.nev'-File directly from the Blackrock Neuro-Signal-Prozessor (via Software)"""""
        nev_data = dict()

        path2file = data_dict['NEV'][sel_file]
        if not path2file and not os.path.exists(path2file):
            nev_data["Exits"] = False
        else:
            nev_data["Exits"] = True
            data_in = BlackrockIO(path2file[:-4], nsx_to_load=self.nsx_version)
            data_in = data_in.nev_data['Spikes']

            # --- Extracting infos from nev-file
            timestamp = []
            electrode = []
            waveform = []
            unit = []
            for idx, data in enumerate(data_in[0]):
                unit.append(data_in[1][idx])
                timestamp.append(data[0])
                electrode.append(data[1])

                in1 = [data[4][idx:idx + 2] for idx in range(0, 95, 2)]
                in2 = [int.from_bytes(data0, 'little', signed=True) for idx, data0 in enumerate(in1)]
                waveform.append(in2)

            electrode = np.array(electrode, dtype=np.int8)
            timestamp = np.array(timestamp)
            waveform = np.array(waveform, dtype=np.int16)
            unit = np.array(unit)

            # --- Set into right format
            no_electrode = np.unique(electrode)
            no_activities = np.zeros(shape=(self.noElectrodes, ), dtype=int)

            for idx, sel_elec in enumerate(no_electrode):
                content = dict()
                selX = np.where(electrode == sel_elec)
                content['timestamps'] = timestamp[selX[0]]
                content['cluster'] = unit[selX[0]]
                content['waveform'] = waveform[selX[0]]
                no_activities[sel_elec-1] = len(selX[0])

                elec_str = 'Elec' + sel_elec.astype('str')
                nev_data[elec_str] = content

            nev_data['NoSpikeActivities'] = np.sum(no_activities)
            nev_data['NoCluster'] = np.unique(unit)

        return nev_data

    def process_label_mat(self, data_dict: dict, sel_file: int) -> dict:
        """ Process '#.mat'-File from the experiment in order to extract frames and positions from Blackrock Signal Processor"""
        # (same name like rawdata and not the behaviour content)
        groundtruth_blackrock = dict()

        path2file = data_dict['MAT'][sel_file]
        if not os.path.exists(path2file):
            groundtruth_blackrock['Exits'] = False
        else:
            groundtruth_blackrock['Exits'] = True
            try:
                groundtruth_old = mat73_loadmat(path2file)
                groundtruth_old = groundtruth_old['NEV']['Data']['Spikes']
            except TypeError:
                groundtruth_old = scipy_loadmat(path2file)

            # --- Processing of data
            timestamp = groundtruth_old['TimeStamp']
            spike_cluster = groundtruth_old['Unit']
            spike_channel = groundtruth_old['Electrode']
            groundtruth_blackrock['Electrode'] = np.unique(spike_channel)
            noElectrode = 96
            no_activities = np.zeros(shape=(noElectrode,), dtype=int)

            for idx in groundtruth_blackrock['Electrode']:
                content = dict()
                selX = np.where(spike_channel == idx)
                content['timestamps'] = timestamp[selX[0]]
                content['cluster'] = spike_cluster[selX[0]]
                content['waveform'] = groundtruth_old['Waveform'][:, selX[0]]
                no_activities[idx - 1] = len(selX[0])

                elec_channel = 'Elec' + idx.astype('str')
                groundtruth_blackrock[elec_channel] = content

            groundtruth_blackrock['NoSpikeActivities'] = np.sum(no_activities)
            groundtruth_blackrock['NoCluster'] = np.unique(spike_cluster)

        return groundtruth_blackrock

    def process_behaviour(self, data_dict: dict) -> dict:
        """Processing the behaviour task of the experiments"""
        behaviour = dict()

        if not data_dict['Exits_BEHV']:
            behaviour['Exits'] = False
        else:
            path2file = data_dict['Behaviour'][0]

            if not path2file and not os.path.exists(path2file):
                print("... Behaviour file is not available")
            else:
                data_behav = scipy_loadmat(path2file)
                behaviour.update({'saveData': data_behav['saveData']})

            return behaviour

    def save_results(self, name: str, rawdata: dict, label: dict, behaviour: dict):
        file_name = os.path.join(self.__path2save, name) + '_MERGED.mat'

        savemat(file_name, {'rawdata': rawdata, 'nev_detected': label, 'behaviour': behaviour})
        print(f"... saved results in: {file_name}")


if __name__ == '__main__':
    path = 'D:'
    folder_name = '10_Klaes_Caltech'
    transfer_alldata = False

    print(f"\nProcessing the datasets of KlaesLab (US, Caltech)")
    klaes_data = UtahDataHandler(path, folder_name)
    if not transfer_alldata:
        sel_folder = [18]
        # sel_folder = range(19, klaes_data.noDataSets)
    else:
        sel_folder = range(0, klaes_data.noDataSets)

    for idx, folder in enumerate(sel_folder):
        data_save = klaes_data.search_data(folder)

        for file in range(0, klaes_data.noDataPoints):
            neural_data = klaes_data.process_rawdata(data_save, file)
            neural_label = klaes_data.process_label_nev(data_save, file)
            # neural_label1 = klaes_data.process_label_mat(data_save, file)
            neural_behaviour = klaes_data.process_behaviour(data_save)

            klaes_data.save_results(
                name=data_save['Name'][file],
                rawdata=neural_data,
                label=neural_label,
                behaviour=neural_behaviour
            )