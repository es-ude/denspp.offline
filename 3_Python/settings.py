import numpy as np

class Settings:
    # SETTINGS ABOUT DATA SOURCE
    path2data = "C:/HomeOffice/Arbeit/B_MERCUR_SpAIke/6_Daten"
    load_data_set = 1
    load_data_point = 1
    t_range = np.array([0])  # Angabe des zu betrachteten Zeitfensters [Start, Ende] in sec.
    ch_sel = -1  # Auswahl der Elektroden(= -1, ruft alle Daten auf)

    # SETTINGS ABOUT FRAMEWORK
    version = 1  # Version of SpikeSorting Pipeline (0: normal - 1: with DAE)

    # --- Properties of the analogue front-end
    # Supply Voltage
    udd = 0.6
    uss = -0.6
    # Analogue pre-amplification
    fs_ana = 50e3  # Neuabtastungs-Rate der Eingangsdaten
    gain_ana = 40
    n_filt_ana = 4
    f_filt_ana = np.array([100, 8e3])
    delay_ana = 100e-6
    # ADC
    oversampling = 1
    n_bit_adc = 12
    d_uref = 0.1
    fs_adc = 20e3
    # 20e3 für 40 samples per frame
    # 32e3 for 64 samples per frame

    # --- Digital filtering for ADC output and CIC
    gain_dig = 1
    n_filt_dig = 2
    f_filt_dig = np.array([100, 6e3])
    delay_dig = 1e-3

    # --- Properties of spike detection and frame generation/aligning
    d_xsda = np.array([1])
    mode_thres = 3
    mode_frame = 1
    x_window_mean = 100
    x_window_length = 2e-3
    x_window_start = 0.7e-3
    x_offset = 0.5e-3

    # --- Properties for feature extraction and clustering
    no_cluster = 2
