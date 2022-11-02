import numpy as np

from settings import Settings
from scipy.io import loadmat

# ----- Read Settings -----
class Labeling:
    exist = 1
    orig_cluster_id = None
    orig_no_cluster = None
    orig_xpos_spike = None
    orig_no_spike = None


def call_data(type, desired_fs, t_range):
    labeling = Labeling()
    # ----- Read data input -----
    mat_data = loadmat("dataset/01_SimDaten_Martinez2009/simulation_1.mat")
    chan = mat_data["chan"]
    data = mat_data["data"]
    sampling_interval = mat_data["samplingInterval"]
    spike_class = mat_data["spike_class"]
    spike_times = mat_data["spike_times"]
    start_data = mat_data["startData"]

    orig_fs = 1e3 / sampling_interval[0, 0]
    gain_pre = 10 ** (0 / 20)
    no_electrodes = 1
    type_mea = "Synthetic"
    labeling.exist = 1
    labeling.orig_cluster_id = spike_class[0, 0]
    labeling.orig_no_cluster = np.unique(spike_class[0]).size
    labeling.orig_xpos_spike = spike_times[0, 0]
    labeling.orig_no_spike = spike_times[0].size
    uin0 = 0.5e-6 * data / gain_pre
    t0 = np.arange(0, data.size - 1, 1).reshape((1, data.size - 1)) / orig_fs
    length_data_orig = t0.size
    if t_range.size == 2:
        t0_idxs = np.argwhere(t_range[0, 0] <= t0)
        t1_idxs = np.argwhere(t_range[0, 1] <= t0)
        if t0_idxs.size > 0 and t1_idxs.size >0:
            t0_idx = t0_idxs[0]
            t0_idx = t1_idxs[0]
            t0 = to[]

# ----- Preparation : Module calling -----
print("hi")
# afe = AFE(
#     afe_set,
# )
# ----- Preparation : Variable declaration -----

# ----- Calculation -----


# ----- Real time state adjustment -----

# ----- Analog Front End Module  -----

# ----- Feature Extraction and Classification Module -----

# ----- After Processing for each Channel -----

# ----- Determination of quality of Parameters -----

# ----- Figures -----
