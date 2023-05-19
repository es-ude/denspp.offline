from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat

from pipeline.pipeline_nev import Settings, Pipeline
from src.data_call import DataController
from src.plotting import results_fec, results_ivt, results_firing_rate

if __name__ == "__main__":
    plt.close('all')
    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = '{}_pipeline'.format(str_datum)
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024) - NEV content")

    # --- Loading the data
    settings = Settings()
    datahand = DataController(settings.SettingsDATA)
    datahand.do_call()
    dataIn = datahand.get_data()
    dataWave = datahand.nev_waveform

    # --- Import mat-file for feature vector
    # feat_in = loadmat("src_decoder/data.mat")
    # feat_vec = feat_in['Data']
    # plot_featvec(feat_vec)

    # --- Processing the data
    SpikeSorting = Pipeline(settings)
    for elec in range(0, dataIn.noChannel):
        SpikeSorting.run(dataWave[elec])
        # --- Plotting
        path2save = SpikeSorting.saving_results(folder_name)
        results_fec(SpikeSorting, path2save, elec)
        results_ivt(SpikeSorting, path2save, elec)
        results_firing_rate(SpikeSorting, path2save, elec)
        plt.show(block=False)

    # --- Ending
    print("\nThe End")
