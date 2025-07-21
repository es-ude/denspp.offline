import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.pyspice_load import PySpiceLoad, SettingsPySpice
from denspp.offline.analog.pyspice_handler import create_dummy_signal


test_sets = SettingsPySpice(
    type='R',
    fs_ana=10e3,
    noise_en=False,
    params_use={'r': 1e3},
    temp_kelvin=300,
    input_volt=True
)


class TestPySpiceLoad(TestCase):
    def test_get_version(self):
        set0 = deepcopy(test_sets)
        version = PySpiceLoad(set0).get_ngspice_version()
        chck = 'PySpice v1.5 with NGSpice v34'
        self.assertEqual(version, chck)

    def test_print_circuits(self):
        set0 = deepcopy(test_sets)
        version = PySpiceLoad(set0).print_spice_circuit()
        chck = '.title Test\r\n\r\n'
        self.assertEqual(version, chck)

    def test_print_types(self):
        set0 = deepcopy(test_sets)
        type_device = PySpiceLoad(set0).print_types()
        assert type_device[0] == 'R = Resistor Circuit (params = [\'r\'])'
        self.assertTrue(len(type_device) > 0)

    def test_dc_simulation_single_resistor(self):
        set0 = deepcopy(test_sets)
        val = PySpiceLoad(set0).do_dc_simulation(1.0)
        assert val['v_in'] == np.array(1.0)
        self.assertEqual(val['i_in'], np.array(1e-3))

    def test_dc_simulation_sweep_resistor(self):
        set0 = deepcopy(test_sets)
        val = PySpiceLoad(set0).do_dc_sweep_simulation(-1., 1., 0.5)
        np.testing.assert_almost_equal(val['i_in'], np.array([-0.001, -0.0005, 0., 0.0005, 0.001]), decimal=6)

    def test_ac_simulation_voltage_divider(self):
        set0 = deepcopy(test_sets)
        set0.type = 'VD'
        set0.params_use = {'r0': 1e3, 'r1': 1e3, 'rl': 1e12, 'cl': 0.0}
        val = PySpiceLoad(set0).do_ac_simulation(1, 100, 2)
        np.testing.assert_almost_equal(val['v_out'], np.abs(np.array([0.5, 0.5, 0.5, 0.5, 0.5])), decimal=6)

    def test_transient_sinusoidal_voltage(self):
        set0 = deepcopy(test_sets)
        set0.type = 'VD'
        set0.params_use = {'r0': 1e3, 'r1': 1e3, 'rl': 1e12, 'cl': 0.0}
        set0.input_volt = True
        val = PySpiceLoad(set0).do_transient_sinusoidal_simulation(
            amp=1.0,
            freq=set0.fs_ana / 50,
            t_sim=100 / set0.fs_ana,
            t_dly=0.0,
            offset=0.0
        )
        scale = (set0.params_use['r0'] + set0.params_use['r1']) / set0.params_use['r1']
        np.testing.assert_almost_equal(val['v_in'], scale * val['v_out'], decimal=6)

    def test_transient_pulse_voltage(self):
        set0 = deepcopy(test_sets)
        set0.type = 'VD'
        set0.params_use = {'r0': 1e3, 'r1': 1e3, 'rl': 1e12, 'cl': 0.0}
        set0.input_volt = True
        val = PySpiceLoad(set0).do_transient_pulse_simulation(
            neg_value=-1.0,
            pos_value=+1.0,
            pulse_width=10/set0.fs_ana,
            pulse_period=20/set0.fs_ana,
            t_sim=100/set0.fs_ana
        )
        scale = (set0.params_use['r0'] + set0.params_use['r1']) / set0.params_use['r1']
        np.testing.assert_almost_equal(val['v_in'], scale*val['v_out'], decimal=6)

    def test_transient_arbitrary_voltage(self):
        set0 = deepcopy(test_sets)
        set0.type = 'VD'
        set0.params_use = {'r0': 1e3, 'r1': 1e3, 'rl': 1e12, 'cl': 0.0}
        set0.input_volt = True
        time, sig = create_dummy_signal(100 / set0.fs_ana, set0.fs_ana, freq_used=[set0.fs_ana / 20], freq_amp=[1.])
        val = PySpiceLoad(set0).do_transient_arbitrary_simulation(sig, time[-1], set0.fs_ana)

        scale = (set0.params_use['r0'] + set0.params_use['r1']) / set0.params_use['r1']
        np.testing.assert_almost_equal(val['v_in'], scale*val['v_out'], decimal=6)

if __name__ == '__main__':
    main()
