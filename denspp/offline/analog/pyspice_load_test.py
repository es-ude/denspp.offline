import numpy as np
import pytest

from denspp.offline.analog.pyspice_handler import create_dummy_signal
from denspp.offline.analog.pyspice_load import PySpiceLoad, SettingsPySpice


@pytest.fixture
def test_sets():
    return SettingsPySpice(
        type="R",
        fs_ana=10e3,
        noise_en=False,
        params_use={"r": 1e3},
        temp_kelvin=300,
        input_volt=True,
    )


@pytest.mark.simulation
def test_get_version(test_sets):
    version = PySpiceLoad(test_sets).get_ngspice_version()
    chck = "PySpice v1.5 with NGSpice v34"
    assert version == chck


@pytest.mark.simulation
def test_print_circuits(test_sets):
    version = PySpiceLoad(test_sets).print_spice_circuit()
    chck = ".title Test\r\n\r\n"
    assert version == chck


@pytest.mark.simulation
def test_print_types(test_sets):
    type_device = PySpiceLoad(test_sets).print_types()
    assert type_device[0] == "R = Resistor Circuit (params = ['r'])"
    assert len(type_device) > 0


@pytest.mark.simulation
def test_dc_simulation_single_resistor(test_sets):
    val = PySpiceLoad(test_sets).do_dc_simulation(1.0)
    assert val["v_in"] == np.array(1.0)
    assert val["i_in"] == np.array(1e-3)


@pytest.mark.simulation
def test_dc_simulation_sweep_resistor(test_sets):
    val = PySpiceLoad(test_sets).do_dc_sweep_simulation(-1.0, 1.0, 0.5)
    np.testing.assert_almost_equal(
        val["i_in"], np.array([-0.001, -0.0005, 0.0, 0.0005, 0.001]), decimal=6
    )


@pytest.mark.simulation
def test_ac_simulation_voltage_divider(test_sets):
    test_sets.type = "VD"
    test_sets.params_use = {"r0": 1e3, "r1": 1e3, "rl": 1e12, "cl": 0.0}
    val = PySpiceLoad(test_sets).do_ac_simulation(1, 100, 2)
    np.testing.assert_almost_equal(val["v_out"], np.abs(np.array([0.5, 0.5, 0.5, 0.5, 0.5])), decimal=6)


@pytest.mark.simulation
def test_transient_sinusoidal_voltage(test_sets):
    test_sets.type = "VD"
    test_sets.params_use = {"r0": 1e3, "r1": 1e3, "rl": 1e12, "cl": 0.0}
    test_sets.input_volt = True
    val = PySpiceLoad(test_sets).do_transient_sinusoidal_simulation(
        amp=1.0,
        freq=test_sets.fs_ana / 50,
        t_sim=100 / test_sets.fs_ana,
        t_dly=0.0,
        offset=0.0,
    )
    scale = (test_sets.params_use["r0"] + test_sets.params_use["r1"]) / test_sets.params_use["r1"]
    np.testing.assert_almost_equal(val["v_in"], scale * val["v_out"], decimal=6)


@pytest.mark.simulation
def test_transient_pulse_voltage(test_sets):
    test_sets.type = "VD"
    test_sets.params_use = {"r0": 1e3, "r1": 1e3, "rl": 1e12, "cl": 0.0}
    test_sets.input_volt = True
    val = PySpiceLoad(test_sets).do_transient_pulse_simulation(
        neg_value=-1.0,
        pos_value=+1.0,
        pulse_width=10 / test_sets.fs_ana,
        pulse_period=20 / test_sets.fs_ana,
        t_sim=100 / test_sets.fs_ana,
    )
    scale = (test_sets.params_use["r0"] + test_sets.params_use["r1"]) / test_sets.params_use["r1"]
    np.testing.assert_almost_equal(val["v_in"], scale * val["v_out"], decimal=6)


@pytest.mark.simulation
def test_transient_arbitrary_voltage(test_sets):
    test_sets.type = "VD"
    test_sets.params_use = {"r0": 1e3, "r1": 1e3, "rl": 1e12, "cl": 0.0}
    test_sets.input_volt = True
    time, sig = create_dummy_signal(
        100 / test_sets.fs_ana, test_sets.fs_ana, freq_used=[test_sets.fs_ana / 20], freq_amp=[1.0]
    )
    val = PySpiceLoad(test_sets).do_transient_arbitrary_simulation(sig, time[-1], test_sets.fs_ana)

    scale = (test_sets.params_use["r0"] + test_sets.params_use["r1"]) / test_sets.params_use["r1"]
    np.testing.assert_almost_equal(val["v_in"], scale * val["v_out"], decimal=6)


if __name__ == "__main__":
    pytest.main([__file__])
