import numpy as np

import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *

from charge_balancer import *


logger = Logging.setup_logging()

# --circuit parameters--
R_tis = 20@u_kOhm
R_far = 10@u_MOhm
C_dl = 6@u_nF
v_cm = 1@u_mV

# --simulation parameters--
n_rep = 10
i_amp=12e-6
f_samp=10e6
f_samp_wfg=200*f_samp
t_start=1e-3
t_end=1e-3
t_sine=2e-3
n_waves=2

# --charge balancer (includes waveform generation)--

cb_offset = CBSettings(
    cbstrat=CBStrat.OFFSET_CURRENT,
    adaptive=False,
    window=0.1,
    adjust=0.005*i_amp
)

cb_dclipping = CBSettings(
    cbstrat=CBStrat.ANO_DCLIPPING,
    adaptive=False,
    window=0.1,
    adjust=0.05*t_sine
)

cb_duration = CBSettings(
    cbstrat=CBStrat.ANO_DURATION,
    adaptive=False,
    window=0.1,
    adjust=0.2*t_sine
)

cb_amplitude = CBSettings(
    cbstrat=CBStrat.ANO_AMPLITUDE,
    adaptive=False,
    window=0.1,
    adjust=0.2
)

cb_aclipping = CBSettings(
    cbstrat=CBStrat.ANO_ACLIPPING,
    adaptive=False,
    window=0.1,
    adjust=0.2*i_amp
)

wfgsettings = WFGSettings(
    i_amp=i_amp,
    f_samp=f_samp,
    f_samp_wfg=f_samp_wfg,
    t_start=t_start,
    t_end=t_end,
    t_sine=t_sine,
    n_waves=n_waves
)

cbal = ChargeBalancer(wfgsettings=wfgsettings, cbsettings=cb_aclipping)

# --current stimulator--
class SinePulseStimulator(NgSpiceShared):
    def __init__(self, waveform, f_samp_wfg, **kwargs):
        super().__init__(**kwargs)
        self._waveform = waveform
        self._f_samp_wfg = float(f_samp_wfg)

    def update_waveform(self, waveform):
        self._waveform = waveform

    def get_vsrc_data(self, voltage, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_vsrc_data @{} node {}'.format(ngspice_id, time, node))
        voltage[0] = self._waveform[int(time*float(self._f_samp_wfg)) % len(self._waveform)]
        return 0

    def get_isrc_data(self, current, time, node, ngspice_id):
        self._logger.debug('ngspice_id-{} get_isrc_data @{} node {}'.format(ngspice_id, time, node))
        current[0] = self._waveform[int(time*float(self._f_samp_wfg)) % len(self._waveform)]
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

# --create simulator--
sine_stim = SinePulseStimulator(waveform=cbal.waveform,
                                f_samp_wfg=cbal.f_samp_wfg, send_data=False)
simulator = circuit.simulator(temperature=25, nominal_temperature=25, 
                              simulator='ngspice-shared', ngspice_shared=sine_stim)

for rep in range(n_rep):
    sine_stim.update_waveform(cbal.waveform)
    
    simulator.initial_condition(point=prev_v_point)
    analysis_i = simulator.transient(step_time=1/cbal.f_samp, end_time=cbal.get_t_rep())
    time = np.concatenate((time, analysis_i.time + start_time))
    i_in = np.concatenate((i_in, analysis_i.vistim_plus))
    i_cdl = np.concatenate((i_cdl, analysis_i.vcdl_plus))
    i_rfar = np.concatenate((i_rfar, analysis_i.vrfar_plus))
    v_out = np.concatenate((v_out, analysis_i.out_p - analysis_i.out_n))
    v_cdl = np.concatenate((v_cdl, analysis_i.point - analysis_i.out_n))
    v_rtis = np.concatenate((v_rtis, analysis_i.out_p - analysis_i.point))    
    start_time += analysis_i.time[-1]
    prev_v_point = analysis_i.point[-1]

    cbal.perform_charge_balancing(float(prev_v_point))

    print(rep, prev_v_point)

# --graph plotting--
figure1, [ax1, ax2] = plt.subplots(2, 1)
ax1.set_title('Currents')
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Current [A]')
ax1.grid()
ax1.plot(time, i_in, label=r'$I_{in}(t)$')
ax1.plot(time, i_rfar, label=r'$I_{Rfar}(t)$')
ax1.plot(time, i_cdl, label=r'$I_{Cdl}(t)$')
ax1.legend()

ax2.set_title('Voltages')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Voltage [V]')
ax2.grid()
ax2.plot(time, v_out, label=r'$V_{out}(t)$')
ax2.plot(time, v_cdl, label=r'$V_{Cdl}(t)$')
ax2.plot(time, v_rtis, label=r'$V_{Rtis}(t)$')
ax2.legend()

plt.tight_layout()
plt.show()

