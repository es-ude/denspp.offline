from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.pyspice_handler import PySpiceHandler, SettingsPySpice


test_sets = SettingsPySpice(
    type='R',
    fs_ana=10e3,
    noise_en=False,
    params_use={'r': 100e3},
    temp_kelvin=300,
    input_volt=True
)


class TestPySpiceHandler(TestCase):
    def test_get_version(self):
        set0 = deepcopy(test_sets)
        version = PySpiceHandler(set0).get_ngspice_version()
        chck = 'PySpice v1.5 with NGSpice v34'
        self.assertEqual(version, chck)

    def test_print_circuits(self):
        set0 = deepcopy(test_sets)
        version = PySpiceHandler(set0).print_spice_circuit()
        chck = '.title Test\r\n\r\n'
        self.assertEqual(version, chck)

    def test_print_types(self):
        set0 = deepcopy(test_sets)
        version = PySpiceHandler(set0).print_types()
        self.assertTrue(version == [])


if __name__ == '__main__':
    main()
