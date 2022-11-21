from settings import Settings
from src.call_data import call_data
from src.afe import AFE
from src.sorting import FEC
import src.plotting as pltSpAIke
from src.setup_gui import setup as gui

class sorting_signals:
    u_in = None
    u_lfp = None
    u_spk = None
    x_adc = None
    x_sda = None
    x_thr = None
    x_pos = None
    frames_orig = None
    frames_align = None
    features = None
    cluster_id = None
    spike_ticks = None

if __name__ == "__main__":
    #gui()
    print("Running spike-sorting frame-work (MERCUR-project Sp:AI:ke, 2022-2024)")
    # ----- Preparation : Module calling -----
    settings = Settings()
    (neuron, labeling) = call_data(
        settings.path2data,
        settings.load_data_set,
        settings.load_data_point,
        settings.desired_fs,
        settings.t_range,
    )
    settings.ch_to_no = neuron.rawdata.shape[0]
    print("... dataset is loaded")
    afe = AFE(settings)
    dfe = FEC(settings)

    # ----- Preparation : Variable declaration -----
    afe_signals = sorting_signals()
    afe_signals.u_in = neuron.data

    if settings.ch_sel == 0:
        settings.ch_sel = settings.ch_to_no

    # ----- Channel Calculation -----
    print("... performing spike sorting incl. analogue pre-processing")
    for idx in range(settings.ch_sel):
        if(afe.realtime_mode):
            # Control logicals for [doADC, doFrame, doAlign]
            # TODO: Realtime mode implementieren
            doCalc = [1, 1, 1]
        else:
            doCalc = [1, 1, 1]

        u_in = afe_signals.u_in[[idx], :]

        # ----- Analogue Front End Module  -----
        u_spk, u_lfp = afe.pre_amp(u_in)
        x_adc = afe.adc_nyquist(u_spk, doCalc[0])

        # --- Digital Pre-processing ----
        x_filt = afe.dig_filt(x_adc)

        # --- Spike detection incl. thresholding
        x_sda, x_thr = afe.spike_detection(x_filt, settings.mode_thres, doCalc[1])
        x_dly = afe.time_delay_dig(x_filt)
        frames_orig, x_pos = afe.frame_generation(x_dly, x_sda, x_thr)
        frames_align = afe.frame_aligning(frames_orig, settings.mode_frame, doCalc[2])

        # ----- Feature Extraction and Classification Module -----
        features = dfe.fe_pca(frames_align)
        (cluster, sse) = dfe.clustering(features)
        spike_ticks = dfe.calc_spiketicks(x_adc, x_pos, cluster)

        # ---- Neural decoder

        # ----- After Processing for each Channel -----
        afe_signals.u_spk = u_spk
        afe_signals.u_lfp = u_lfp
        afe_signals.x_adc = x_adc
        afe_signals.x_sda = x_sda
        afe_signals.x_thr = x_thr

        afe_signals.x_pos = x_pos
        afe_signals.frames_orig = frames_orig
        afe_signals.frames_align = frames_align
        afe_signals.features = features
        afe_signals.cluster_id = cluster
        afe_signals.spike_ticks = spike_ticks

        # ----- Determination of quality of Parameters -----
        if labeling.exist:
            ...
            #quality_param = QualityParam()
            #result_sda = afe.analyze_sda(x_pos, labeling.adc_x_pos_spike, 100)

            #quality_param.dr = result_sda.tpr * result_sda.accuracy
            #quality_param.ca = result_sda.accuracy
            #quality_param.dr = u_spk.size / frames_align.size

    # ----- Figures -----
    print("... plotting results")
    pltSpAIke.resultsAFE(afe.sample_rate_ana, afe.sample_rate_adc, afe_signals)
    pltSpAIke.resultsFEC(afe_signals)

    print("This is the End, ... my only friend, ... the end")
