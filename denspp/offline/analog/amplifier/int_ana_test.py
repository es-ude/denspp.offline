import pytest
import numpy as np
from unittest import TestCase, main
from copy import deepcopy
from .int_ana import SettingsINT, IntegratorAmplifier


TestSettingsINT = SettingsINT(
    vdd=0.6, vss=-0.6,
    fs_ana=1e3,
    tau=100e-3,
    res_in=10e3,
    offset_v=1e-3,
    offset_i=1e-9,
    do_invert=False,
    noise_en=False,
    noise_edev=10e-9
)

@pytest.mark.parametrize("vss, vdd, vcm, vrange", [
    (-1.2, 1.2, 0.0, 2.4),
    (0.0, 1.8, 0.9, 1.8)
])
def test_settings_vcm(vss: float, vdd: float, vcm: float, vrange: float) -> None:
    set0 = deepcopy(TestSettingsINT)
    set0.vss = vss
    set0.vdd = vdd
    assert set0.vcm == vcm
    assert set0.u_supply_range == vrange


class IntegratorAmplifierTest(TestCase):
    dut = IntegratorAmplifier(TestSettingsINT)
    sig_zero = np.zeros(shape=(100,)) + TestSettingsINT.vcm
    sig_ones = sig_zero + 0.1
    sig_sym = np.concatenate((sig_ones[0:50], -sig_ones[50:]), axis=0)

    def test_ideal_integrator_sample_zeros(self):
        chck = TestSettingsINT.vcm
        rslt = self.dut.do_ideal_integration_sample(
            u_inp=self.sig_zero,
            u_inn=TestSettingsINT.vcm
        )
        self.assertEqual(rslt, chck)

    def test_ideal_integrator_sample_ones(self):
        chck = 0.1
        rslt = self.dut.do_ideal_integration_sample(
            u_inp=self.sig_ones,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt)-chck), 1e-6)

    def test_ideal_integrator_sample_sym(self):
        chck = TestSettingsINT.vcm
        rslt = self.dut.do_ideal_integration_sample(
            u_inp=self.sig_sym,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt) - chck), 1e-4)

    def test_ideal_integrator_sample_clamp_vdd(self):
        chck = TestSettingsINT.vdd
        rslt = self.dut.do_ideal_integration_sample(
            u_inp=self.sig_ones * 10,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt) - chck), 1e-4)

    def test_ideal_integrator_sample_clamp_vss(self):
        chck = TestSettingsINT.vss
        rslt = self.dut.do_ideal_integration_sample(
            u_inp=-self.sig_ones * 10,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt)-chck), 1e-4)

    def test_ideal_integrator_signal_zeros(self):
        chck = self.sig_zero
        rslt = self.dut.do_ideal_integration(
            u_inp=self.sig_zero,
            u_inn=TestSettingsINT.vcm
        )
        np.testing.assert_almost_equal(rslt, chck, decimal=5)

    def test_ideal_integrator_signal_ones(self):
        chck = np.linspace(start=0.0, stop=0.1, num=self.sig_ones.size, endpoint=True)
        rslt = self.dut.do_ideal_integration(
            u_inp=self.sig_ones,
            u_inn=TestSettingsINT.vcm
        )
        np.testing.assert_almost_equal(rslt, chck, decimal=2)

    def test_ideal_integrator_clamp_vdd(self):
        chck = TestSettingsINT.vdd
        rslt = self.dut.do_ideal_integration(
            u_inp=self.sig_ones * 10,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt[-1]) - chck), 1e-4)

    def test_ideal_integrator_clamp_vss(self):
        chck = TestSettingsINT.vss
        rslt = self.dut.do_ideal_integration(
            u_inp=-self.sig_ones * 10,
            u_inn=TestSettingsINT.vcm
        )
        self.assertLess(abs(float(rslt[-1]) - chck), 1e-4)

    def test_opa_volt_integrator_zeros(self):
        chck = np.zeros_like(self.sig_zero) + TestSettingsINT.vcm - TestSettingsINT.u_error
        rslt = self.dut.do_opa_volt_integration(
            u_inp=-self.sig_zero,
            u_inn=TestSettingsINT.vcm + TestSettingsINT.u_error
        )
        self.assertLess(float(sum(np.abs(rslt - chck)))/self.sig_zero.size, 1e-4)

    def test_opa_volt_integrator_ones(self):
        chck = -np.linspace(start=0.0, stop=0.1, num=self.sig_ones.size,
                            endpoint=True) + TestSettingsINT.vcm - TestSettingsINT.u_error
        rslt = self.dut.do_opa_volt_integration(
            u_inp=-self.sig_ones,
            u_inn=TestSettingsINT.vcm + TestSettingsINT.u_error
        )
        self.assertLess(float(sum(np.abs(rslt - chck))) / self.sig_ones.size, 1e-3)

    def test_opa_curr_integrator_zeros(self):
        chck = -np.linspace(start=0.0, stop=0.1, num=self.sig_ones.size, endpoint=True)
        chck += TestSettingsINT.vcm - TestSettingsINT.u_error
        chck /= TestSettingsINT.res_in

        rslt = self.dut.do_opa_curr_integration(
            i_in=-self.sig_zero / TestSettingsINT.res_in,
            u_ref=TestSettingsINT.vcm
        )
        self.assertLess(float(sum(np.abs(rslt - chck))) / self.sig_zero.size, 1e-4)

    def test_opa_curr_integrator_ones(self):
        chck = np.zeros_like(self.sig_zero)
        chck += TestSettingsINT.vcm - TestSettingsINT.u_error
        chck /= TestSettingsINT.res_in

        rslt = self.dut.do_opa_curr_integration(
            i_in=-self.sig_ones / TestSettingsINT.res_in,
            u_ref=TestSettingsINT.vcm
        )
        self.assertLess(float(sum(np.abs(rslt - chck))) / self.sig_ones.size, 1e-3)

    def test_cap_curr_integrator_zeros(self):
        chck = np.zeros_like(self.sig_zero)
        chck += TestSettingsINT.vcm + TestSettingsINT.vss

        rslt = self.dut.do_cap_curr_integration(
            i_in=-self.sig_zero / TestSettingsINT.res_in
        )
        self.assertLess(float(sum(np.abs(rslt - chck)))/self.sig_zero.size, 1e-4)

    def test_cap_curr_integrator_ones(self):
        chck = np.linspace(start=0.0, stop=0.1, num=self.sig_ones.size, endpoint=True) * TestSettingsINT.tau_active_scale
        chck += TestSettingsINT.vcm + TestSettingsINT.vss

        rslt = self.dut.do_cap_curr_integration(
            i_in=self.sig_ones / TestSettingsINT.res_in
        )
        self.assertLess(float(sum(np.abs(rslt - chck))) / self.sig_ones.size, 1e-3)


if __name__ == '__main__':
    main()
