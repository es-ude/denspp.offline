import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src_emg.pipeline_emg import Settings, Pipeline
from src_emg.call_emg import DataLoader
from src_emg.plotting_emg import results_input

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning EMG detection")

    # ----- Preparation: Module calling -----
    settings = Settings()
    datahand = DataLoader(settings.SettingsDATA)
    datahand.do_call()
    #datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()
    del datahand
    num_channels = len(dataIn.data_raw)

    # --- Pipeline
    pipe_emg = Pipeline(settings)

    signals_out = [np.zeros((1,), dtype=float) for idx in range(num_channels)]
    for idx, thr in enumerate(tqdm(dataIn.data_raw, ncols=100, desc='Progress: ')):
        pipe_emg.run(thr)
        signals_out[idx] = pipe_emg.signal.x_env

    # ----- Plotting
    results_input(dataIn.data_raw, dataIn.data_fs_orig)
    results_input(signals_out, dataIn.data_fs_orig, dataIn.evnt_xpos, dataIn.evnt_cluster_id)
    plt.show()

    # ----- Ending -----
    print("This is the End, ... my only friend, ... the end")
