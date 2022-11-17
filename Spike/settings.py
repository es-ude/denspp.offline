from pydantic import BaseModel
import numpy as np


class Settings(BaseModel):
    # SETTINGS ABOUT FRAMEWORK
    # Path2Data = "C:/HomeOffice/Austausch/Daten"
    path2data = "dataset"
    load_data_set = 1
    load_data_point = 1

    realtime_mode = 0  #  Realtime - Mode(0: offline, 1: online)
    ch_sel = 0  # Auswahl der Elektroden(= 0, ruft alle Daten auf)
    desired_fs = 100e3  # Neuabtastungs - Rate der Eingangsdaten
    t_range = np.array([[10, 40]])  # Angabe des zu betrachteten Zeit fensters [StartEnde] in sec.
    ch_to_no = 0

    # SETTINGS ABOUT FRAMEWORK
    udd = 0.6  # Supply Voltage
    uss = -0.6

    gain_pre = 25  # midband gain of preamplifier
    n_filt_ana = 2
    f_filt_ana = np.array([[200, 5e3]])

    # --- PropertiesofADC
    oversampling = 1
    n_bit_adc = 12
    d_uref = 0.05
    sample_rate = 30e3

    # --- Digital filtering for ADC output and CIC
    n_filt_dig = 2
    f_filt_dig = np.array([[200, 5e3]])

    # --- Properties of spike detection
    d_xsda = np.array([[2, 4, 6]])
    thres_Mode = 7
    sda_thr_min = 1.5e-9  # (only for mode = 1)

    # --- Properties of Framing and Aligning of spike frames
    x_offset = round(0.5e-3 * sample_rate)
    x_delta_neg = round(0.5e-3 * sample_rate)
    input_delay = x_offset
    x_window_length = round(2e-3 * sample_rate)
    # --- Properties for Labeling data
    no_cluster = 19

    class Config:
        validate_all = False
        validate_assignment = False
        arbitrary_types_allowed = True
