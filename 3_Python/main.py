from datetime import datetime
import matplotlib.pyplot as plt

from pipeline.pipeline_v1 import Settings, Pipeline
from src.data_call import DataController
from src.plotting import results_afe, results_fec, results_paper

if __name__ == "__main__":
    plt.close('all')
    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder_name = '{}_pipeline'.format(str_datum)
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Preparation: Module calling -----
    settings = Settings()
    datahand = DataController(settings.SettingsDATA)
    datahand.do_call()
    datahand.do_cut()
    datahand.do_resample()
    datahand.output_meta()
    dataIn = datahand.get_data()

    # ----- Module declaration & Channel Calculation -----
    SpikeSorting = Pipeline(settings)
    for idx, uin in enumerate(dataIn.raw_data):
        no_electrode = dataIn.channel[idx]
        print(f"\nPerforming end-to-end pipeline on channel {no_electrode}")
        SpikeSorting.run(uin)
        # SpikeSorting.check_label(dataIn)

        # ----- Figures -----
        print("... plotting and saving results")
        path2save = SpikeSorting.saving_results(folder_name)
        results_afe(SpikeSorting, path2save, no_electrode)
        results_fec(SpikeSorting, path2save, no_electrode)
        results_paper(SpikeSorting, path2save, no_electrode)
        plt.show(block=False)

    # ----- Ending -----
    print("This is the End, ... my only friend, ... the end")
