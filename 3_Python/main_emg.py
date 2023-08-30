from datetime import datetime
import matplotlib.pyplot as plt

from pipeline.pipeline_v1 import Settings, Pipeline
from src.metric import Metric
from src.data_call_emg import DataController
from src.plotting import results_afe1, results_fec, results_paper, results_ivt, results_firing_rate, results_correlogram

if __name__ == "__main__":
    plt.close('all')
    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = '{}_pipeline'.format(str_datum)
    print("\nRunning EMG detection")

    # ----- Preparation: Module calling -----
    settings = Settings()
    datahand = DataController(settings.SettingsDATA)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()

    # ----- Module declaration & Channel Calculation -----


    # ----- Ending -----
    print("This is the End, ... my only friend, ... the end")
