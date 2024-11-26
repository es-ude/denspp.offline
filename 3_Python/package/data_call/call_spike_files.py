from os.path import join
import numpy as np
from scipy.io import loadmat
from mat73 import loadmat as loadmat_mat73
from package.data_call.call_cellbib import CellSelector
from package.data_call.call_handler import _DataController, DataHandler, SettingsDATA, translate_unit_to_scale_value


class DataLoader(_DataController):
    """Class for loading and manipulating the used dataset"""
    raw_data: DataHandler
    settings: SettingsDATA
    path2file: str

    def __init__(self, setting: SettingsDATA) -> None:
        _DataController.__init__(self)
        self.settings = setting
        self.select_electrodes = list()
        self.path2file = str()

    def do_call(self):
        """Loading the dataset"""
        self._prepare_call()
        # --- Searching the load function for dataset translation
        methods_list_all = [method for method in dir(DataLoader)]
        search_param = '_DataLoader'
        methods_load_data = [method for method in methods_list_all if method[0:len(search_param)] == search_param]

        # --- Calling the function
        if self.settings.data_set == 0:
            raise ValueError("\nPlease select new input for data_type!")
        else:
            getattr(self, methods_load_data[self.settings.data_set - 1])()

    def __load_method00_martinez2009(self) -> None:
        """Loading synthethic files from Quiroga simulation (2009)"""
        folder_name = "_SimDaten_Martinez2009"
        data_type = 'simulation_*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file)

        self.raw_data = DataHandler()
        # Meta information
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_fs_orig = int(1 / loaded_data["samplingInterval"][0][0] * 1000)
        self.raw_data.device_id = [0]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Input
        self.raw_data.electrode_id = [int(loaded_data["chan"][0])-1]
        self.raw_data.data_raw = [0.5e-6 * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"][0].size / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = True
        spike_xoffset = int(-0.1e-3 * self.raw_data.data_fs_orig)
        self.raw_data.evnt_xpos = [(loaded_data["spike_times"][0][0][0] - spike_xoffset)]
        self.raw_data.evnt_id = [(loaded_data["spike_class"][0][0][0])]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method01_pedreira2012(self) -> None:
        """Loading synthethic files from Quiroga simulator (2012)"""
        folder_name = "_SimDaten_Pedreira2012"
        data_type = 'simulation_*.mat'
        self._prepare_access_file(folder_name, data_type)

        prep_index = self.path2file.split("_")[-1]
        num_index = int(prep_index[0:2])
        path2label = join(self.settings.path, folder_name, "ground_truth.mat")

        loaded_data = loadmat(self.path2file)
        ground_truth = loadmat(path2label)

        self.raw_data = DataHandler()
        # Meta information
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_fs_orig = 24e3
        self.raw_data.device_id = [0]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Raw data
        self.raw_data.electrode_id = [int(loaded_data["data"].shape[0])-1]
        self.raw_data.data_raw = [25e-6 * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"].shape[1] / self.raw_data.data_fs_orig
        # Groundtruth
        self.raw_data.label_exist = True
        spike_xoffset = int(-0.1e-6 * self.raw_data.data_fs_orig)
        self.raw_data.evnt_xpos = [(ground_truth["spike_first_sample"][0][num_index - 1][0] - spike_xoffset)]
        self.raw_data.evnt_id = [(ground_truth["spike_classes"][0][num_index - 1][0])]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method02_quiroga2020(self) -> None:
        """Loading synthetic recordings from Quiroga simulator (Common benchmark)"""
        folder_name = "_SimDaten_Quiroga2020"
        data_type = 'C_*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file, mat_dtype=True)

        self.raw_data = DataHandler()
        # --- Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Synthetic"
        self.raw_data.data_fs_orig = float(1000 / loaded_data["samplingInterval"][0][0])
        self.raw_data.device_id = [0]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Input data
        self.raw_data.electrode_id = [int(loaded_data["chan"][0][0])-1]
        self.raw_data.data_raw = [100e-6 * np.float32(loaded_data["data"][0])]
        self.raw_data.data_time = loaded_data["data"].shape[1] / self.raw_data.data_fs_orig
        # --- Groundtruth
        self.raw_data.label_exist = True
        spike_xoffset = int(-0.5e-6 * self.raw_data.data_fs_orig)
        self.raw_data.evnt_xpos = [(loaded_data["spike_times"][0][0][0] - spike_xoffset)]
        self.raw_data.evnt_id = [(loaded_data["spike_class"][0][0][0] - 1)]
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method03_seidl2012(self) -> None:
        """Loading the recording files from the Freiburg probes from Karsten Seidl from this PhD"""
        folder_name = "_Freiburg_Seidl2014"
        data_type = '*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file)

        self.raw_data = DataHandler()
        # Meta information
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Penetrating"
        self.raw_data.data_lsb = 1 / loaded_data['GainPre'][0][0]
        self.raw_data.data_fs_orig = loaded_data['origFs'][0][0]
        self.raw_data.device_id = [0]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Raw data
        elec_orig = np.arange(0, loaded_data['raw_data'].shape[0]).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for elec in elec_process:
            self.raw_data.data_raw.append(self.raw_data.data_lsb * np.float32(loaded_data['raw_data'][elec]))
        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = loaded_data['raw_data'].shape[1] / self.raw_data.data_fs_orig
        # --- Groundtruth
        self.raw_data.label_exist = False
        spike_xoffset = int(0 * self.raw_data.data_fs_orig)
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method04_marre2018(self) -> None:
        # Link to data: https://zenodo.org/record/1205233#.YrBYrOzP1PZ
        folder_name = "_Zenodo_Marre2018"
        data_type = '*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file)

        self.raw_data = DataHandler()
        # Input and Meta information
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Intracellular Matching"
        self.raw_data.data_fs_orig = int(loaded_data['fs'][0])
        self.raw_data.device_id = [0]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Raw data
        # Information for data structure(Channel No = 255: Müll, No = 254: Intracellular response)
        elec_orig = np.arange(0, loaded_data['juxta_channel'][0]).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for elec in elec_process:
            self.raw_data.data_raw.append(float(loaded_data['Gain'][0]) * (np.float32(loaded_data['data'][:, elec]) - 32767))
        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = loaded_data['data'].shape[0] / self.raw_data.data_fs_orig
        # --- Groundtruth
        self.raw_data.label_exist = False
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method05_klaes_utah_array(self) -> None:
        """Loading the merged data file (from *.ns6 and *.nev files) from recordings with Utah electrode array
        from Blackrock Neurotechnology"""
        folder_name = "_Klaes_Caltech"
        data_type = '*_MERGED.mat'
        nsp_device = self.settings.data_set
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file, mat_dtype=True)

        self.raw_data = DataHandler()
        # Input and meta
        gain_str = loaded_data['rawdata']['LSB'][0, 0][0][0:-1].split(" ")
        data_lsb = translate_unit_to_scale_value(gain_str, 1) * float(gain_str[0])

        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Utah"
        self.raw_data.data_fs_orig = int(loaded_data['rawdata']['SamplingRate'][0, 0][0])
        self.raw_data.device_id = [nsp_device]
        # Electrode mapping information
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [10, 10]
        # Raw data
        elec_orig = np.arange(0, int(loaded_data['rawdata']['NoElectrodes'][0, 0][0])).tolist()
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        data_raw = np.transpose(loaded_data['rawdata']['spike'][0, 0])
        for elec in elec_process:
            self.raw_data.data_raw.append(data_lsb * np.float32(data_raw[elec]))
        self.raw_data.electrode_id = elec_process
        self.raw_data.data_time = data_raw.shape[0]
        # --- Groundtruth from BlackRock
        self.raw_data.label_exist = int(loaded_data['nev_detected']['Exits'][0, 0][0])
        spike_xoffset = int(-0.1e-6 * self.raw_data.data_fs_orig)
        for elec in elec_process:
            data = loaded_data['nev_detected'][f'Elec{1 + elec}'][0, 0]
            self.raw_data.evnt_xpos.append(data['timestamps'][0, 0][0, :] - spike_xoffset)
            self.raw_data.evnt_id.append(data['cluster'][0, 0][0, :])

        # --- Behaviour
        self.raw_data.behaviour_exist = True
        self.raw_data.behaviour = loaded_data['behaviour']
        del loaded_data

    def __load_method06_rgc_tdb(self) -> None:
        """Loading the transient files from the Retinal Ganglion Cell Transient Database (RGC TDB)"""
        folder_name = "_RGC_TDB"
        data_type = '*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat_mat73(self.path2file)

        # Pre-Processing: Remove empty entries and runs with only one spike
        check_xpos = loaded_data['sp_trains']['sp']
        check_data = loaded_data['sp_trains']['data']
        used_ch = list()
        for idx, pos in enumerate(check_xpos):
            if not isinstance(pos[0], str) and pos[0] is not None and check_data[idx][0] is not None:
                if pos[0].ndim == 1:
                    used_ch.append(idx)
        del check_data, check_xpos

        # Pre-Processing: Getting only the desired channels
        elec_orig = used_ch
        if not len(self.select_electrodes) == 0:
            elec_process = list()
            for elec in self.select_electrodes:
                elec_process.append(elec_orig[elec])
        else:
            elec_process = elec_orig

        spike_xpos = list()
        data_raw = list()
        for elec in elec_process:
            spike_xpos.append(loaded_data['sp_trains']['sp'][elec][0].astype('int'))
            data_raw.append(loaded_data['sp_trains']['data'][elec][0])

        self.raw_data = DataHandler()
        # Input and meta --- This type has only one channel. Used for experiment runs
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "Isolated RGC"
        self.raw_data.data_fs_orig = int(loaded_data['sp_trains']['sample_rate'][0][0])
        self.raw_data.device_id = [0]
        # Electrode Mapping
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [1, 1]
        # Raw data
        self.raw_data.electrode_id = np.arange(0, len(elec_process)).tolist()
        for idx, pos_ch in enumerate(elec_process):
            self.raw_data.data_raw.append(1e-6 * np.float32(data_raw[idx]-data_raw[idx][0]))
        self.raw_data.data_time = self.raw_data.data_raw[0].shape[0] / self.raw_data.data_fs_orig
        # Groundtruth
        spike_xoffset = int(-0.5e-6 * self.raw_data.data_fs_orig)
        rgc_translator = CellSelector(1, 0)
        self.raw_data.label_exist = True
        for idx, pos_ch in enumerate(elec_process):
            self.raw_data.evnt_xpos.append(spike_xpos[idx] - spike_xoffset)
            num_spikes = len(spike_xpos[idx])
            type = loaded_data['sp_trains']['cell_type'][pos_ch][0]
            id = rgc_translator.get_id_from_celltype(type)
            if id == -1:
                print(f"Missing type: {type}")
            self.raw_data.evnt_id.append(np.zeros(shape=(num_spikes,), dtype=int) + id)

        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method07_fzj_mcs(self) -> None:
        """Loading the recording files from MCS setup in FZ Juelich (case = experiment, point = file)"""
        folder_name = "_RGC_FZJuelich"
        data_type = '*_merged.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat_mat73(self.path2file)

        self.raw_data = DataHandler()
        # Input and meta
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "MCS 60MEA"
        self.raw_data.data_fs_orig = float(loaded_data['fs'])
        self.raw_data.device_id = [0]
        # Electrode Mapping
        self.raw_data.mapping_exist = False
        self.raw_data.mapping_dimension = [8, 8]
        # Raw data
        elec_orig = [int(val[2:]) for val in loaded_data["head_name"]]
        elec_process = self.select_electrodes if not len(self.select_electrodes) == 0 else elec_orig
        for idx, elec in enumerate(elec_process):
            self.raw_data.data_raw.append(float(loaded_data['gain']) * np.float32(loaded_data['electrode'][:, idx]))
        self.raw_data.data_time = loaded_data['electrode'].shape[0] / self.raw_data.data_fs_orig
        self.raw_data.electrode_id = elec_process
        # Groundtruth
        self.raw_data.label_exist = False
        # Behaviour
        self.raw_data.behaviour_exist = False
        self.raw_data.behaviour = None
        del loaded_data

    def __load_method08_musall_neuropixel(self) -> None:
        """Loading the files from recordings with NeuroPixel probes"""
        folder_name = "_RGC_TDB"
        data_type = '*.mat'
        self._prepare_access_file(folder_name, data_type)
        loaded_data = loadmat(self.path2file)

        self.raw_data = DataHandler()
        self.raw_data.data_name = folder_name
        self.raw_data.data_type = "NeuroPixel 1.0"

        self.raw_data.mapping_dimension = [2, 480]
        del loaded_data

        raise NotImplementedError
