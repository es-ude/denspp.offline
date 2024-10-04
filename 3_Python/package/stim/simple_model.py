import matplotlib.pyplot as plt

import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

from package.data_generator.waveform_generator import WaveformGenerator

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
n_rep = 5

cdl_offset = 0

# --waveform generator--
if not float(t_rest):
    time_points       = [0,              float(t_sine) *3/4]
    time_duration     = [float(t_sine),  float(t_rep - t_sine)]
    time_wfg          = [5,              13]
    polarity_cathodic = [True,           False]
else:
    time_points       = [0,              float(t_pulse), float(t_pulse + t_rest), float(t_pulse * 2 + t_rest)        ]
    time_duration     = [float(t_pulse), float(t_rest),  float(t_pulse),          float(t_rep - t_pulse * 2 - t_rest)]
    time_wfg          = [3,              13,             3,                       13                                  ]
    polarity_cathodic = [True,           False,          False,                   False]

# print(int(t_rep/t_samp_wfg))
wfg = WaveformGenerator(float(f_samp_wfg))
t, waveform = wfg.generate_waveform(time_points, time_duration, time_wfg, polarity_cathodic)
# t = t[0:int(t_rep/t_samp_wfg)]
# waveform = waveform[0:int(t_rep/t_samp_wfg)]
# waveform[0] = 0

# print(t.size)

# prev = 0
# for i in range(len(t)-1): # time steps are variable during sim
#     curr = t[i+1] - t[i]
#     if curr != prev:
#         prev = curr
#         print(i, ': ', curr)

# print(waveform[int(0.001*f_samp_wfg)])

# --check waveform--
plt.figure()
plt.plot(t, waveform, 'k')
plt.xlabel("Time t / s")
plt.ylabel("Signal y(t)")
plt.grid()
plt.tight_layout()
plt.show()

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

# --simulation--
sine_stim = SinePulseStimulator(waveform=waveform, amplitude=i_amp, 
                                f_samp_wfg=f_samp_wfg, send_data=False)
simulator = circuit.simulator(temperature=25, nominal_temperature=25, 
                              simulator='ngspice-shared', ngspice_shared=sine_stim)
# initial condition for "floating" capacitor (no parallel resistor)
simulator.initial_condition(point=v_cm + cdl_offset)
analysis = simulator.transient(step_time=t_samp, end_time=t_rep*n_rep)

# print(analysis.time.as_ndarray()[:20]) # repeated zeros at first few time points
# print(analysis.vistim_plus.as_ndarray()[:20])

# print(float(t_rep)*n_rep)
# print(analysis.vrfar_plus[0]) # init value is non-zero at non-zero init conditions
# print(analysis.vcdl_plus[0]) # init value is zero at non-zero init conditions
# print(analysis.point[0]) # init value is not exact to init condition set by user, might mean time has elapsed
# print(analysis.out_n[0])
# print(len(analysis.time)) # 100008 samples
# prev = 0
# for i in range(len(analysis.time)-1): # time steps are variable during sim
#     curr = analysis.time[i+1] - analysis.time[i]
#     if curr != prev:
#         prev = curr
#         print(i, ': ', curr)
# print(analysis.time.as_ndarray()[-1])
# print(analysis.time.as_ndarray())
# print(analysis.time.as_ndarray()[-1] - analysis.time.as_ndarray()[-2])
# print(analysis.time.as_ndarray()[1] - analysis.time.as_ndarray()[0])
# print(f_samp)

# print(analysis.vrload_plus.as_ndarray())

# # --check if waveform is charge-balanced--
# print('waveform_generator:')
# wfg.check_charge_balancing(waveform)
# print('current source created:')
# wfg.check_charge_balancing(analysis.vistim_plus.as_ndarray())

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

# # charge = integrate.cumulative_trapezoid(analysis.vcdl_plus, analysis.time, initial=0)

# # ax3.set_title('Injected Charge')
# # ax3.set_xlabel('Time [s]')
# # ax3.set_ylabel('Charge [C]')
# # ax3.grid()
# # ax3.plot(analysis.time, charge)

# # ax.plot(analysis.time, analysis.out_p - analysis.out_n)
# # ax.plot(analysis.time, analysis.vistim_plus)
# # ax.plot(analysis.time[0:4000], analysis.vistim_plus.as_ndarray()[0:4000] - waveform * float(i_amp))
# # ax.plot(analysis.time, analysis.point - analysis.out_n)
# # for node in analysis.branches.values():
# #     ax.plot(analysis.time, node)

# plt.tight_layout()
# plt.show()


