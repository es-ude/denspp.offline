import matplotlib.pyplot as plt

from pipeline.pipeline_nev import Settings, Pipeline
from src_decoder.plotting import plot_featvec
from src.data_call import DataController
from scipy.io import loadmat

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings = Settings()
    datahand = DataController(settings.SettingsDATA)
    datahand.do_call()
    dataIn = datahand.get_data()

    # --- Import mat-file for feature vector
    feat_in = loadmat("src_decoder/data.mat")
    feat_vec = feat_in['Data']

    # --- Feature Mapping

    # --- Plotting
    plot_featvec(feat_vec)

    # --- Ending
    print("\nThe End")
