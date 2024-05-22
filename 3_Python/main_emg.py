import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src_emg.call_emg import DataLoader, SettingsDATA
from src_emg.plotting_emg import results_input
from src_emg.pipeline_emg import Pipeline

SettingsDATA = SettingsDATA(
        path='C:/HomeOffice/Data_EMG',
        data_set=1,
        data_case=0,
        data_point=1,
        t_range=[0],
        ch_sel=[-1],
        fs_resample=1e3
    )

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning EMG detection")

    # ----- Preparation: Module calling -----
    datahand = DataLoader(SettingsDATA)
    datahand.do_call()
    #datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand
    num_channels = len(dataIn.data_raw)

    # --- Pipeline
    pipe_emg = Pipeline(SettingsDATA.fs_resample)

    # --- Processing
    signals_out = [dict() for idx in range(num_channels)]
    for idx, thr in enumerate(tqdm(dataIn.data_raw, ncols=100, desc='Progress: ')):
        pipe_emg.run(thr)
        signals_out[idx] = pipe_emg.signal.x_sda

    # ----- Plotting
    results_input(dataIn.data_raw, dataIn.data_fs_orig,
                  path2save=pipe_emg.path2save, addon='_input')
    results_input(signals_out, dataIn.data_fs_orig, dataIn.evnt_xpos, dataIn.evnt_cluster_id,
                  path2save=pipe_emg.path2save, addon='_after')
    plt.show()

    # ----- Ending -----
    print("This is the End, ... my only friend, ... the end")
