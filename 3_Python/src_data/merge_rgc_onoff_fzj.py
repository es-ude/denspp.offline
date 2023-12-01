from mat73 import loadmat
from scipy.io import savemat
from os.path import join
import numpy as np
from glob import glob
from datetime import datetime


if __name__ == "__main__":
    frames_in = list()
    frames_cl = list()
    cluster_dict = list()

    path2folder = "D:/0_Invasive/07_RGC_ONOFF_FZJ/Original"
    folders = glob(join(path2folder, 'waveforms_*.mat'))

    # --- Getting the data
    for idx, folder in enumerate(folders):
        print(f"Processing file #{1+idx}: {folder}")
        data = loadmat(folder)
        data = data['tot_spike_matrix']

        for frame in data:
            x0 = 40-12
            x1 = x0 + 40
            frames_in.append(frame[x0:x1])
            frames_cl.append(idx)

        cluster_dict.append(folder.split('_')[-1][:-4])

    frames_in = np.array(frames_in, dtype=np.float16)
    frames_cl = np.array(frames_cl, dtype=np.int16)

    # --- Saving the data
    savemat(f'../data/{datetime.now().strftime("%Y-%m-%d")}_rgc_onoff_fzj_Merged.mat',
            {"frames_in": frames_in, "frames_cluster": frames_cl, "cluster_dict": cluster_dict,
             "create_time": datetime.now().strftime("%Y-%m-%d"), "fsamp": 25000},
            do_compression=True, long_field_names=True)
    print("This is the End!")
