from pydantic import BaseModel
import numpy as np

class Settings(BaseModel):
    # SETTINGS ABOUT FRAMEWORK
    path2data = "C:/HomeOffice/Austausch/Daten"
    # path2data = "dataset"
    load_data_set = 1
    load_data_point = 1

    realtime_mode = 0               # Realtime - Mode(0: offline, 1: online)
    ch_sel = 0                      # Auswahl der Elektroden(= 0, ruft alle Daten auf)
    desired_fs = 100e3              # Neuabtastungs-Rate der Eingangsdaten
    t_range = np.array([10, 12])  # Angabe des zu betrachteten Zeit fensters [Start, Ende] in sec.
    ch_to_no = 0

    # SETTINGS ABOUT FRAMEWORK
    udd = 0.6  # Supply Voltage
    uss = -0.6

    # --- Properties of the analogue pre-amplificaiton
    gain_ana = 40
    n_filt_ana = 1
    f_filt_ana = np.array([100, 8e3])
    delay_ana = 100e-6

    # --- Properties of ADC
    oversampling = 1
    n_bit_adc = 12
    d_uref = 0.1
    sample_rate = 20e3

    # --- Digital filtering for ADC output and CIC
    gain_dig = 1
    n_filt_dig = 2
    f_filt_dig = np.array([100, 6e3])
    delay_dig = 0.5e-3

    # --- Properties of spike detection and frame generation/aligning
    d_xsda = np.array([1])
    mode_thres = 1
    mode_frame = 1
    x_window_length = 2e-3
    x_window_start = 0.7e-3
    x_offset = 0.5e-3

    # --- Properties for feature extraction and clustering
    mode_feature = 1
    mode_cluster = 1
    no_cluster = 19

    class Config:
        validate_all = False
        validate_assignment = False
        arbitrary_types_allowed = True
