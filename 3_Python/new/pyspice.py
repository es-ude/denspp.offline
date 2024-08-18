from PySpice.Spice.Netlist import Circuit
import matplotlib.pyplot as plt
import numpy as np


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


def plot_iv_curve(u_in: np.ndarray, i_out: np.ndarray, do_log=False) -> None:
    """"""
    plt.figure()
    if not do_log:
        plt.plot(u_in, i_out)
    else:
        plt.semilogy(u_in, np.abs(i_out))
    plt.xlabel(r'Voltage $U$ / V')
    plt.ylabel(r'Current $I$ / A')

    plt.tight_layout()
    plt.grid()
    plt.savefig('test.svg', format='svg')
    plt.show(block=True)


def plot_transient(time: np.ndarray, v_in: np.ndarray, v_out: np.ndarray, i_in: np.ndarray) -> None:
    """"""
    fig, ax1 = plt.subplots()
    ax1.plot(time, v_in, label='inp')
    ax1.plot(time, v_out, label='out')
    ax1.set_ylabel(r"Voltage $U$ / V")
    ax1.set_xlabel(r'Time $t$ / s')

    ax2 = ax1.twinx()
    ax2.plot(time, i_in, 'r--', label='current')
    ax2.set_ylabel(r"Current $I$ / A")

    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig('test.svg', format='svg')
    plt.show(block=True)


def plot_bodeplot(freq: np.ndarray, v_in: np.ndarray, v_out: np.ndarray) -> None:
    """"""
    transfer_function = v_out / v_in
    fig, ax1 = plt.subplots()
    ax1.semilogx(freq, 20 * np.log10(np.abs(transfer_function)), 'k')
    ax1.set_ylabel(r"Gain $v_U$ / dB")
    ax1.set_xlabel(r'Frequency $f$ / dB')

    ax2 = ax1.twinx()
    ax2.semilogx(freq, np.angle(transfer_function, deg=True), 'r')
    ax2.set_ylabel(r"Phase $\alpha$ / °")

    plt.tight_layout()
    plt.grid()
    plt.savefig('test.svg', format='svg')
    plt.show(block=True)


# --- Easy Tutorial for Using PySpice: https://github.com/benedictjones/engineeringthings-pyspice (Git),
# https://www.youtube.com/watch?v=62BOYx1UCfs&list=PL97KTNA1aBe1QXCcVIbZZ76B2f0Sx2Snh (YouTube)

if __name__ == "__main__":
    # --- Definition of circuit
    mode = 3

    circuit = Circuit('Test')
    circuit.model('myDiode', 'D', IS=4e-12, RS=0, BV=10, IBV=1e-12, N=2, VJ=7, M=0.125)

    # --- Definition of input
    if mode == 0:
        circuit.V('input', 'sense', circuit.gnd, 1)
    elif mode == 1:
        circuit.V('input', 'sense', circuit.gnd, 0)
    elif mode == 2:
        circuit.PulseVoltageSource('input', 'sense', circuit.gnd,
                                   initial_value=-1.8, pulsed_value=1.8,
                                   pulse_width=10e-3, period=20e-3)
    elif mode == 3:
        circuit.SinusoidalVoltageSource('input', 'sense', circuit.gnd, amplitude=1, frequency=100)
    elif mode == 4:
        sine_stim = SinePulseStimulator(waveform=waveform, amplitude=i_amp,
                                        f_samp_wfg=f_samp_wfg, send_data=False)

    # Add terminal for measuring current
    circuit.R(0, 'sense', 'in', 0)
    circuit.R0.plus.add_current_probe(circuit)

    # --- Circuit Definition
    circuit.R(1, 'in', 'out', 10e3)
    circuit.R(2, 'out', 'ref', 10e3)
    circuit.C(1, 'out', 'ref', 100e-9)
    #circuit.Diode(0, 'out', circuit.gnd, model='MyDiode')
    circuit.V('ref', 'ref', circuit.gnd, 0.0)

    print(circuit)

    # --- Definition of simulation
    if mode == 0:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()
        for node in analysis.nodes.values():
            print(f'Node {str(node)}: {float(node):4.3f} V')
    elif mode == 1:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.dc(Vinput=slice(-2, 5, .01))
    elif mode == 2:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.transient(step_time=100e-6, end_time=100e-3)
        time = np.array(analysis.time)
    elif mode == 3:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.ac(start_frequency=1, stop_frequency=1e6, number_of_points=100, variation='dec')
        freq = np.array(analysis.frequency)
    elif mode == 4:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25,
                                      simulator='ngspice-shared', ngspice_shared=sine_stim)
        simulator.initial_condition(point=v_cm + cdl_offset)
        analysis = simulator.transient(step_time=t_samp, end_time=t_rep * n_rep)
    else:
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = 0

    # --- Get values
    i_in = np.array(analysis.branches['vr0_plus'])
    v_in = np.array(analysis['in'])
    v_out = np.array(analysis['out'])

    # --- Plotten
    if mode == 0:
        pass
    elif mode == 1:
        plot_iv_curve(v_in, i_in, do_log=True)
    elif mode == 2:
        plot_transient(time, v_in, v_out, i_in)
    elif mode == 3:
        plot_bodeplot(freq, v_in, v_out)
