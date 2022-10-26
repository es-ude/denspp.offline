from pydantic import BaseModel
import scipy
import os
import numpy as np


class Settings(BaseModel):

    realtime_mode = 0
    #  Realtime - Mode(0: offline, 1: online)
    realtime_mode = 0
    # Auswahl des zu verwendenen Datenquelle (0: Datensatz laden, 1: Alte Konfig.)
    data_type = 1
    # Auswahl der Elektroden( = 0, ruft alle Daten auf)
    ch_sel = 0
    # Neuabtastungs - Rate der Eingangsdaten
    desired_fs = 100e3
    # Angabe des zu betrachteten Zeit fensters [StartEnde] in sec.
    t_range = np.array([10, 40])
    # Using ParallelComputing for MultiChannel - Auslese(0: disable, 1: enable)
    enable_parallel_computing = 0
    # Setting Analog Front End
    # Supply Voltage
    udd = 0.6
    uss = -0.6

    # --- Preamplification and bandpass filtering

    gain_pre = 25
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

    # (only for mode = 1)
    sda_thr_min = 1.5e-9
    # --- Properties of Framing and Aligning of spike frames

    x_offset = round(0.5e-3 * sample_rate)
    x_delta_neg = round(0.5e-3 * sample_rate)
    input_delay = x_offset
    x_window_length = round(2e-3 * sample_rate)
    # --- Properties for Labeling data
    no_cluster = 19
