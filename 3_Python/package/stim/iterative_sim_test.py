import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
from scipy import integrate

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *

from waveform_generator import WaveformGenerator

logger = Logging.setup_logging()

# --circuit parameters--
R_tis = 20@u_kOhm
R_far = 10@u_MOhm
C_dl = 6@u_nF
v_cm = 1@u_mV

# --current stimulator parameters--
i_amp = 12@u_uA
f_samp = 10@u_MHz
f_samp_wfg = 200 * f_samp
t_rep = 4@u_ms
t_sine = 2@u_ms
t_pulse = t_sine / 2

t_samp = f_samp.period
t_samp_wfg = f_samp_wfg.period
f_rep = t_rep.frequency
f_sine = t_sine.frequency
f_pulse = t_pulse.frequency
t_rest = 0
n_rep = 20

# --waveform generator--
if not float(t_rest):
    time_points       = [0,              float(t_sine)        ]
    time_duration     = [float(t_sine), float(t_rep - t_sine)]
    time_wfg          = [5,              13]
    polarity_cathodic = [True,           False]
else:
    time_points       = [0,              float(t_pulse), float(t_pulse + t_rest), float(t_pulse * 2 + t_rest)        ]
    time_duration     = [float(t_pulse), float(t_rest),  float(t_pulse),          float(t_rep - t_pulse * 2 - t_rest)]
    time_wfg          = [3,              13,             3,                       13                                  ]
    polarity_cathodic = [True,           False,          False,                   False]

# print(int(t_rep/t_samp))
wfg = WaveformGenerator(float(f_samp_wfg))
t, waveform = wfg.generate_waveform(time_points, time_duration, time_wfg, polarity_cathodic)
# t = t[0:int(t_rep/t_samp)]
# waveform = waveform[0:int(t_rep/t_samp_wfg)]
# waveform[0] = 0

# prev = 0
# for i in range(len(t)-1): # time steps are variable during sim
#     curr = t[i+1] - t[i]
#     if curr != prev:
#         prev = curr
#         print(i, ': ', curr)

# # --check waveform--
# plt.figure()
# plt.plot(t, waveform, 'k')
# plt.xlabel("Time t / s")
# plt.ylabel("Signal y(t)")
# plt.grid()
# plt.tight_layout()
# plt.show()

# --current stimulator--
class SinePulseStimulator(NgSpiceShared):
    def __init__(self, waveform, amplitude, f_samp_wfg, **kwargs):
        super().__init__(**kwargs)
        self._waveform = waveform
        self._amplitude = float(amplitude)
        self._f_samp_wfg = float(f_samp_wfg)

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        voltage[0] = self._amplitude * self._waveform[int(time*float(self._f_samp_wfg)) % len(self._waveform)]
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
        current[0] = self._amplitude * self._waveform[int(time*float(self._f_samp_wfg)) % len(self._waveform)]
        return 0

# --circuit design--
circuit = Circuit('SimpleCurrentControlledStimulation')

circuit.I('stim', circuit.gnd, 'out_p', 'dc 0 external')
circuit.R('tis', 'out_p', 'point', R_tis)
circuit.R('far', 'point', 'out_n', R_far)
circuit.C('dl', 'point', 'out_n', C_dl)
circuit.V('cm', 'out_n', circuit.gnd, v_cm)

circuit.Istim.plus.add_current_probe(circuit)
circuit.Rtis.plus.add_current_probe(circuit)
circuit.Rfar.plus.add_current_probe(circuit)
circuit.Cdl.plus.add_current_probe(circuit)

# print(str(circuit))

# --create simulator--
sine_stim = SinePulseStimulator(waveform=waveform, amplitude=i_amp, 
                                f_samp_wfg=f_samp_wfg, send_data=False)
simulator = circuit.simulator(temperature=25, nominal_temperature=25, 
                              simulator='ngspice-shared', ngspice_shared=sine_stim)

# --one-off simulation--
simulator.initial_condition(point=v_cm)
analysis = simulator.transient(step_time=t_samp, end_time=t_rep*n_rep)

# print(analysis.time.as_ndarray()[:20])
# print(analysis.vistim_plus.as_ndarray()[:20])

# --iterative simulation--
start_time = 0@u_s
prev_v_point = v_cm
time = []
i_in = []
i_cdl = []
i_rfar = []
v_out = []
v_cdl = []
v_rtis = []

for rep in range(n_rep):
    print(rep)
    simulator.initial_condition(point=prev_v_point)
    analysis_i = simulator.transient(step_time=t_samp, end_time=t_rep)
    time = np.concatenate((time, analysis_i.time + start_time))
    i_in = np.concatenate((i_in, analysis_i.vistim_plus))
    i_cdl = np.concatenate((i_cdl, analysis_i.vcdl_plus))
    i_rfar = np.concatenate((i_rfar, analysis_i.vrfar_plus))
    v_out = np.concatenate((v_out, analysis_i.out_p - analysis_i.out_n))
    v_cdl = np.concatenate((v_cdl, analysis_i.point - analysis_i.out_n))
    v_rtis = np.concatenate((v_rtis, analysis_i.out_p - analysis_i.point))    
    start_time += analysis_i.time[-1]
    prev_v_point = analysis_i.point[-1]

# # --check time--
# print(len(time))
# plt.figure()
# plt.plot(time, 'k')
# plt.grid()
# plt.tight_layout()
# plt.show()

# --comparison--
print(len(analysis.time), len(time))
print(analysis.out_p[-1] - analysis.out_n[-1] - v_out[-1])

# # --graph plotting--
# figure1, [ax1, ax2] = plt.subplots(2, 1)
# ax1.set_title('Currents')
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Current [A]')
# ax1.grid()
# ax1.plot(analysis.time, analysis.vistim_plus, label=r'$I_{in}(t)$')
# ax1.plot(analysis.time, analysis.vrfar_plus, label=r'$I_{Rfar}(t)$')
# ax1.plot(analysis.time, analysis.vcdl_plus, label=r'$I_{Cdl}(t)$')
# ax1.legend()

# ax2.set_title('Voltages')
# ax2.set_xlabel('Time [s]')
# ax2.set_ylabel('Voltage [V]')
# ax2.grid()
# ax2.plot(analysis.time, analysis.out_p - analysis.out_n, label=r'$V_{out}(t)$')
# ax2.plot(analysis.time, analysis.point - analysis.out_n, label=r'$V_{Cdl}(t)$')
# ax2.plot(analysis.time, analysis.out_p - analysis.point, label=r'$V_{Rtis}(t)$')
# ax2.legend()
# plt.tight_layout()

# figure2, [ax1, ax2] = plt.subplots(2, 1)
# ax1.set_title('Currents')
# ax1.set_xlabel('Time [s]')
# ax1.set_ylabel('Current [A]')
# ax1.grid()
# ax1.plot(time, i_in, label=r'$I_{in}(t)$')
# ax1.plot(time, i_rfar, label=r'$I_{Rfar}(t)$')
# ax1.plot(time, i_cdl, label=r'$I_{Cdl}(t)$')
# ax1.legend()

# ax2.set_title('Voltages')
# ax2.set_xlabel('Time [s]')
# ax2.set_ylabel('Voltage [V]')
# ax2.grid()
# ax2.plot(time, v_out, label=r'$V_{out}(t)$')
# ax2.plot(time, v_cdl, label=r'$V_{Cdl}(t)$')
# ax2.plot(time, v_rtis, label=r'$V_{Rtis}(t)$')
# ax2.legend()

# plt.tight_layout()
# plt.show()

