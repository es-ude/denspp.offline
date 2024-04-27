from datetime import datetime
import matplotlib.pyplot as plt

from src_emg.pipeline_emg import Settings
from src_emg.call_emg import DataLoader
from src_emg.plotting_emg import results_input

if __name__ == "__main__":
    plt.close('all')
    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = '{}_pipeline'.format(str_datum)
    print("\nRunning EMG detection")

    # ----- Preparation: Module calling -----
    settings = Settings()
    datahand = DataLoader(settings.SettingsDATA)
    datahand.do_call()
    #datahand.do_cut()
    #datahand.do_resample()
    #datahand.output_meta()
    dataIn = datahand.get_data()

    # ----- Module declaration & Channel Calculation -----

    # ----- Plotting
    results_input(dataIn.raw_data)
    plt.show()

    # ----- Ending -----
    print("This is the End, ... my only friend, ... the end")
