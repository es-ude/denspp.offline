import matplotlib.pyplot as plt

from settings import Settings
from src_decoder.settings import DataSettings, DSPSettings, DecoderSettings
from src_decoder.plotting import plot_featvec
from src.data_call import DataController
from src.dsp.dsp import DSP
from scipy.io import loadmat

if __name__ == "__main__":
    plt.close('all')
    print("\nRunning spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings = DataSettings()
    datahandler = DataController(
        path2data=settings.path2data,
        used_channel=settings.ch_sel
    )
    datahandler.do_call(
        data_type=settings.load_data_set,
        data_set=settings.load_data_point
    )
    datahandler.output_meta()
    dataIn = datahandler.get_data()

    # --- Import mat-file for feature vector
    feat_in = loadmat("src_decoder/data.mat")
    feat_vec = feat_in['Data']

    # --- Filtering the raw data (getting the lfp signal)
    setting_dsp = Settings()
    BP_LFP = DSP(
        setting=setting_dsp,
        f_filt_dig=setting_dsp.f_filt_lfp
    )
    BP_AP = DSP(
        setting=setting_dsp,
        f_filt_dig=setting_dsp.f_filt_spk
    )

    u_ap = list()
    u_lfp = list()
    for idx, uin in enumerate(dataIn.raw_data):
        u_ap.append(BP_AP.dig_filt_iir(uin))
        u_lfp.append(BP_LFP.dig_filt_iir(uin))
        print(f"Process on channel: {dataIn.channel[idx]}")

    # --- Feature Mapping

    # --- Plotting
    plot_featvec(feat_vec)

    # --- Ending
    print("\nThe End")
