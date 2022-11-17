from settings import Settings
from src.call_data import call_data
from src.afe import AFE

import matplotlib.pyplot as plt
import numpy as np


class AFESignals:
    u_in = None
    u_lfp = None
    u_spk = None
    x_adc = None
    x_sda = None
    x_thr = None
    x_trg = None
    x_pos = None
    frames_orig = None
    frames_align = None
    features = None
    cluster_id = None
    spike_ticks = None


class QualityParam:
    dr = None
    ca = None
    cr = None


if __name__ == "__main__":
    print("Running frameworking for spike-sorting (MERCUR-project Sp:AI:ke, 2022-2024)")

    settings = Settings()
    afe_signals = AFESignals()
    quality_param = QualityParam()
    # ----- Preparation : Module calling -----
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

    # ----- Preparation : Variable declaration -----
    afe_signals.u_in = neuron.data
    ch_sel = settings.ch_sel + 1
    afe_signals.u_lfp = [None] * ch_sel
    afe_signals.u_spk = [None] * ch_sel

    afe_signals.x_adc = [None] * ch_sel
    afe_signals.x_sda = [None] * ch_sel
    afe_signals.x_thr = [None] * ch_sel
    afe_signals.x_trg = [None] * ch_sel

    afe_signals.x_pos = [None] * ch_sel
    afe_signals.frames_orig = [None] * ch_sel
    afe_signals.frames_align = [None] * ch_sel
    afe_signals.features = [None] * ch_sel
    afe_signals.cluster_id = [None] * ch_sel
    afe_signals.spike_ticks = [None] * ch_sel

    quality_param.dr = [None] * ch_sel
    quality_param.ca = [None] * ch_sel
    quality_param.cr = [None] * ch_sel

    if settings.ch_sel == 0:
        settings.ch_sel = settings.ch_to_no

    for idx in range(settings.ch_sel):

        # --- Anpassungen für Realtime-Anwendung
        do_adc = 1
        u_in = afe_signals.u_in[[idx], :]

        #  --- Modules of Analogue Front-end
        u_spk, u_lfp = afe.pre_amp(u_in)
        x_adc, _ = afe.adc_nyquist(u_spk, do_adc)
        x_dly = afe.time_delay_dig(x_adc)
        x_trg, x_sda, x_thr = afe.spike_detection(x_adc, settings.thres_Mode, do_adc)
        frames_orig, x_pos = afe.frame_generation(x_dly, x_trg)
        frames_align = afe.frame_aligning(frames_orig, 2)

        # --- Modules of Feature Extraction and Classification
        # (only for pre-labeling data input)
        feat_array, feat_cell = afe.fe_pca(frames_align)
        cluster_props = afe.clustring(feat_array)
        spike_ticks = afe.determine_spike_ticks(x_pos, cluster_props, x_adc)

        afe_signals.u_spk[idx] = u_spk
        afe_signals.u_lfp[idx] = u_lfp

        afe_signals.x_adc[idx] = x_adc
        afe_signals.x_sda[idx] = x_sda
        afe_signals.x_thr[idx] = x_thr
        afe_signals.x_trg[idx] = x_trg

        afe_signals.x_pos[idx] = x_pos
        afe_signals.frames_orig[idx] = frames_orig
        afe_signals.frames_align[idx] = frames_align
        afe_signals.features[idx] = feat_array
        afe_signals.cluster_id[idx] = cluster_props
        afe_signals.spike_ticks[idx] = spike_ticks

        if labeling.exist:
            labeling.adc_x_pos_spike = round(labeling.ist_x_pos_spike * settings.sample_rate / settings.desired_fs)
            result_sda = afe.analyze_sda(x_trg, labeling.adc_x_pos_spike, 100)

            quality_param.dr[idx] = result_sda.tpr * result_sda.accuracy
            quality_param.ca[idx] = result_sda.accuracy
            quality_param.dr[idx] = u_spk.size / frames_align.size

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
