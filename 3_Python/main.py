from datetime import datetime
import matplotlib.pyplot as plt

from settings import Settings
import src.plotting as pltSpAIke
from src.pipeline import PipelineSpike as spaike
from src.data_call import DataController

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_logsdir = '{}_pipeline'.format(str_datum)

    # ----- Preparation : Module calling -----
    settings = Settings()
    datahandler = DataController(
        path2data=settings.path2data,
        used_channel=settings.ch_sel
    )
    datahandler.do_call(
        data_type=settings.load_data_set,
        data_set=settings.load_data_point
    )
    datahandler.do_cut(
        t_range=settings.t_range
    )
    datahandler.do_resample(
        desired_fs=settings.fs_ana
    )
    datahandler.output_meta()
    dataIn = datahandler.get_data()

    # ----- Preparation : Variable declaration -----
    SpikeSorting = spaike(settings)
    SpikeSorting.initMod(settings.version)

    # ----- Channel Calculation -----
    for idx, uin in enumerate(dataIn.raw_data):
        print(f"\nPerforming end-to-end pipeline on channel {dataIn.channel[idx]}")
        SpikeSorting.u_in = uin
        SpikeSorting.runPipeline()
        #SpikeSorting.check_label(dataIn)

    # ----- Figures -----
    print("... plotting and saving results")
    logsdir = SpikeSorting.saving_results(experiment_logsdir)
    pltSpAIke.results_afe(SpikeSorting, logsdir)
    pltSpAIke.results_fec(SpikeSorting, logsdir)
    pltSpAIke.results_paper(SpikeSorting, logsdir)
    plt.show(block = False)

    print("This is the End, ... my only friend, ... the end")
