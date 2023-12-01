from datetime import datetime
import matplotlib.pyplot as plt

from src_decoder.pipeline_nev import Settings, Pipeline
from package.data.data_call_common import DataController
from package.plotting.plot_pipeline import results_fec, results_ivt, results_firing_rate

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
    elec_id = dataIn.electrode_id
    del dataIn
    dataWave = datahand.nev_waveform

    # --- Processing the data
    SpikeSorting = Pipeline(settings)
    path2save = SpikeSorting.saving_results(folder_name)
    
    for elec in dataIn.electrode_id:
        SpikeSorting.run(dataWave[elec])
        # --- Plotting
        results_fec(SpikeSorting, elec, path2save)
        results_ivt(SpikeSorting, elec, path2save)
        results_firing_rate(SpikeSorting, elec, path2save)
        plt.show(block=False)

    # --- Ending
    print("\nThe End")
