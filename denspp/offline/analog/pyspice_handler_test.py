import pytest

from denspp.offline.analog.pyspice_handler import PySpiceHandler, SettingsPySpice


@pytest.fixture
def test_sets():
    return SettingsPySpice(
        type="R",
        fs_ana=10e3,
        noise_en=False,
        params_use={"r": 100e3},
        temp_kelvin=300,
        input_volt=True,
    )


@pytest.mark.simulation
def test_get_version(test_sets):
    version = PySpiceHandler(test_sets).get_ngspice_version()
    chck = "PySpice v1.5 with NGSpice v34"
    assert version == chck


@pytest.mark.simulation
def test_print_circuits(test_sets):
    version = PySpiceHandler(test_sets).print_spice_circuit()
    chck = ".title Test\r\n\r\n"
    assert version == chck


@pytest.mark.simulation
def test_print_types(test_sets):
    version = PySpiceHandler(test_sets).print_types()
    assert version == []


if __name__ == "__main__":
    pytest.main([__file__])
