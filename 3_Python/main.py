import numpy as np
import matplotlib.pyplot as plt
from settings import Settings

import src.plotting as pltSpAIke
from src.pipeline import PipelineSpike as spaike
from src.call_data import call_data

if __name__ == "__main__":
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    # ----- Preparation : Module calling -----
    settings = Settings()
    (neuron, labeling) = call_data(
        path2data=settings.path2data,
        data_type=settings.load_data_set,
        data_set=settings.load_data_point,
        desired_fs=settings.desired_fs,
        t_range=settings.t_range,
        ch_sel=settings.ch_sel,
        plot=True
    )
    print("... dataset is loaded")

    # ----- Preparation : Variable declaration -----
    SpikeSorting = spaike(settings)
    SpikeSorting.initMod(settings.version)

    # ----- Channel Calculation -----
    print("... performing spike sorting incl. analogue pre-processing")
    for idx in range(0, 1):
        if settings.realtime_mode:
            # TODO: Realtime-Mode implementieren
            # Control logical for [doADC, doFrame, doAlign]
            doCalc = [1, 1, 1]
        else:
            doCalc = [1, 1, 1]

        SpikeSorting.u_in = neuron.data
        SpikeSorting.runPipeline(doCalc)

        # ----- Determination of quality of Parameters -----
        if labeling.exist:
            xposIst = np.round(settings.sample_rate/neuron.orig_fs * labeling.spike_xpos)
            SpikeSorting.metric_afe(xposIst)

    # ----- Figures -----
    print("... plotting results")
    pltSpAIke.results_afe(SpikeSorting)
    pltSpAIke.results_fec(SpikeSorting)
    plt.show(block = True)

    print("This is the End, ... my only friend, ... the end")
