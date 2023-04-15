import numpy as np
import matplotlib.pyplot as plt
from settings import Settings

import src.plotting as pltSpAIke
from src.pipeline import PipelineSpike as spaike
from src.call_data import DataController

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    # ----- Preparation : Module calling -----
    settings = Settings()
    datahandler = DataController(
        path2data=settings.path2data,
        sel_channel=settings.ch_sel
    )
    datahandler.do_call(
        data_type=settings.load_data_set,
        data_set=settings.load_data_point
    )
    datahandler.do_resample(
        t_range=settings.t_range,
        desired_fs=settings.fs_ana
    )
    datahandler.output_meta()
    dataIn = datahandler.get_data()

    # ----- Preparation : Variable declaration -----
    SpikeSorting = spaike(settings)
    SpikeSorting.initMod(settings.version)

    # ----- Channel Calculation -----
    print("... performing spike sorting incl. analogue pre-processing")
    for idx in range(0, 1):
        SpikeSorting.u_in = dataIn.raw_data
        SpikeSorting.runPipeline()

        # ----- Determination of quality of Parameters -----
        if dataIn.label_exist:
            xposIst = np.round(settings.fs_adc/settings.fs_ana * dataIn.spike_xpos)
            SpikeSorting.metric_afe(xposIst)

    # ----- Figures -----
    print("... plotting results")
    pltSpAIke.results_afe(SpikeSorting)
    pltSpAIke.results_fec(SpikeSorting)
    plt.show(block = True)

    print("This is the End, ... my only friend, ... the end")
