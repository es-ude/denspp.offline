from settings import Settings
from src.call_data import call_data
from src.afe import AFE

import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("Running frameworking for spike-sorting (MERCUR-project Sp:AI:ke, 2022-2024)")

    Settings = Settings()
    # ----- Preparation : Module calling -----
    (neuron, labeling) = call_data(
        Settings.Path2Data, Settings.LoadDataSet, Settings.LoadDataPoint, Settings.desired_fs, Settings.t_range
    )
    print("... dataset is loaded")
    afe = AFE(Settings)

    # ----- Preparation : Variable declaration -----

    # ----- Calculation -----

    # ----- Real time state adjustment -----

    # ----- Analog Front End Module  -----

    # ----- Feature Extraction and Classification Module -----

    # ----- After Processing for each Channel -----

    # ----- Determination of quality of Parameters -----

    # ----- Figures -----
    print("... plotting results")
    plt.figure(1)
    plt.plot(neuron.time, 1e6 * neuron.data)
    plt.xlabel("Time t / s")
    plt.ylabel("Input voltage $U_{in}$ / µV")
    plt.show()
