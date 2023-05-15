import numpy as np

class DataSettings:
    """"Settings for Loading and Handling different data sets"""
    # SETTINGS ABOUT DATA SOURCE
    path2data = "C:\HomeOffice\Arbeit\C_MERCUR_SpAIke\Daten"
    load_data_set = 6
    load_data_point = 0
    # Angabe des zu betrachteten Zeitfensters [Start, Ende] in sec.
    t_range = np.array([0])
    # Auswahl der Elektroden(= -1, ruft alle Daten auf)
    # ch_sel = -1
    ch_sel = [34, 55, 95]

class DSPSettings:
    """"Settings for Digital Signal Processing """
    fs_adc = 20e3
    gain_dig = 1
    n_filt_dig = 2
    f_filt_lfp = np.array([0.1, 100])
    f_filt_spk = np.array([100, 6e3])

class DecoderSettings:
    """Settings for Neural Decoding"""
    pass