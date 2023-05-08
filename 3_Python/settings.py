import numpy as np

class Settings:
    # SETTINGS ABOUT DATA SOURCE
    path2data = "C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten"
    load_data_set = 6
    load_data_point = 0
    # Angabe des zu betrachteten Zeitfensters [Start, Ende] in sec.
    t_range = np.array([0])
    # Auswahl der Elektroden(= -1, ruft alle Daten auf)
    # ch_sel = -1
    ch_sel = [34, 55, 95]

    # SETTINGS ABOUT FRAMEWORK
    # Version of SpikeSorting Pipeline (0: normal - 1: with DAE)
    version = 0

    # --- Properties of the analogue front-end
    # Supply Voltage
    udd = 0.6
    uss = -0.6
    # Analogue pre-amplification
    # Neuabtastungs-Rate der Eingangsdaten
    fs_ana = 50e3
    gain_ana = 40
    n_filt_ana = 1
    f_filt_ana = np.array([0.1, 8e3])

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
    f_filt_lfp = np.array([0.1, 100])
    f_filt_spk = np.array([100, 6e3])

    # --- Properties of spike detection and frame generation/aligning
    # x_offset[0] = time_delay of spike input to frame generation
    dx_sda = np.ndarray([1])
    dx_neo = 1
    dx_mteo = np.array([1, 2, 4, 5])

    mode_thres = 2
    mode_frame = 3
    x_window_mean = 1
    x_window_length = 1.6e-3
    x_window_start = 0.4e-3
    x_offset = [400e-6, 300e-6]

    # --- Properties for feature extraction and clustering
    no_cluster = 3
