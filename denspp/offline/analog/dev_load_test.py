import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.dev_load import ElectricalLoad, SettingsDevice


TestSettings0 = SettingsDevice(
    type='R',
    fs_ana=50e3,
    noise_en=False,
    use_poly=False,
    params_use={'r': 100e3},
    temp=300
)
TestSettings1 = SettingsDevice(
    type='RDs',
    fs_ana=50e3,
    noise_en=False,
    use_poly=False,
    params_use={'i_sat': 1e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3},
    temp=300
)


# --- Info: Function have to start with test_*
class TestElectricalLoad(TestCase):
    voltage = np.linspace(start=0, stop=1, num=21, endpoint=True)

    def test_device_lib_access(self):
        set0 = deepcopy(TestSettings0)
        ovr_devices = ElectricalLoad(settings_dev=set0).get_type_list()
        self.assertGreater(len(ovr_devices), 0)

    def test_param_bounds_list(self):
        set0 = deepcopy(TestSettings1)
        set0.type = 'RDs'
        param_list = ElectricalLoad(settings_dev=TestSettings1).declare_param_bounds_curvefit(
            {'i_sat': [1e-18, 1e-3], 'n_eff': [0.8, 5.0], 'r_sh': [1e3, 100e3], 'uth0': [0.1, 1.6]}
        )
        check = [[1e-18, 0.8, 0.1, 1000.0], [0.001, 5.0, 1.6, 100000.0]]
        self.assertEqual(param_list, check)

    def test_extract_params_from_measurement_for_resistor_wo_bounds(self) -> None:
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.params_use = {'r': 100e3}
        model = ElectricalLoad(settings_dev=TestSettings0)

        current_model = model.get_current(self.voltage, 0.0)
        params = model.extract_params_curvefit(
            voltage=self.voltage,
            current=current_model,
            param_bounds={'r': [0.5 * set0.dev_value['r'], 2 * set0.dev_value['r']]}
        )
        np.testing.assert_almost_equal(set0.dev_value['r'], params['R'.lower()], decimal=1)

    def test_extract_params_from_measurement_for_resistor_with_bounds(self) -> None:
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.params_use = {'r': 100e3}
        model = ElectricalLoad(settings_dev=TestSettings0)

        current_model = model.get_current(self.voltage, 0.0)
        params = model.extract_params_curvefit(
            voltage=self.voltage,
            current=current_model,
            param_bounds={'r': [0.5* set0.dev_value['r'], 2 * set0.dev_value['r']]}
        )
        np.testing.assert_almost_equal(set0.dev_value['r'], params['r'], decimal=1)

    def test_extract_params_from_measurement_for_diode_with_bounds(self) -> None:
        set0 = deepcopy(TestSettings1)
        set0.type = 'RDd'
        set0.params_use = {'i_sat': 1e-12, 'n_eff': 2.8, 'uth0': 0.4, 'r_sh': 20e3}
        model = ElectricalLoad(settings_dev=TestSettings1)

        current_model = model.get_current(self.voltage, 0.0)
        params = model.extract_params_curvefit(
            voltage=self.voltage,
            current=current_model,
            param_bounds={'i_sat': [1e-15, 1e-9], 'n_eff': [0.8, 5.0], 'r_sh': [1e3, 100e3], 'uth0': [0.3, 0.5]}
        )

        result_current = model._get_current_from_equation(self.voltage, 0.0, params)
        check_current = current_model
        np.testing.assert_almost_equal(result_current, check_current, decimal=3)

    def test_resistor_get_voltage_normal(self):
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.noise_en = False
        set0.use_poly = False
        set0.params_use = {'r': 100e3}
        curr = ElectricalLoad(settings_dev=set0).get_current(self.voltage, 0.0)
        np.testing.assert_almost_equal(curr * set0.params_use['r'], self.voltage, decimal=8)

    def test_resistor_get_voltage_poly(self):
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.noise_en = False
        set0.use_poly = True
        set0.params_use = {'r': 100e3}

        curr = ElectricalLoad(settings_dev=set0).get_current(self.voltage, 0.0)
        np.testing.assert_almost_equal(curr * set0.params_use['r'], self.voltage, decimal=8)

    def test_resistor_get_current_normal(self):
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.noise_en = False
        set0.use_poly = False
        set0.params_use = {'r': 100e3}
        volt = ElectricalLoad(settings_dev=set0).get_voltage(self.voltage / set0.params_use['r'], 0.0)
        np.testing.assert_almost_equal(volt, self.voltage, decimal=8)

    def test_resistor_get_current_poly(self):
        set0 = deepcopy(TestSettings0)
        set0.type = 'R'
        set0.noise_en = False
        set0.use_poly = True
        set0.params_use = {'r': 100e3}
        volt = ElectricalLoad(settings_dev=set0).get_voltage(self.voltage/set0.params_use['r'], 0.0)
        np.testing.assert_almost_equal(volt, self.voltage, decimal=8)

    def test_resistive_diode_get_current_normal(self):
        set0 = deepcopy(TestSettings0)
        set0.type = 'RDs'
        set0.noise_en = False
        set0.use_poly = False
        set0.params_use = {'i_sat': 10e-12, 'n_eff': 2.8, 'uth0': 0.1, 'r_sh': 20e3}
        curr = ElectricalLoad(settings_dev=set0).get_current(self.voltage, 0.0)
        self.assertGreater(curr.min(), 1e-14)


if __name__ == '__main__':
    main()
