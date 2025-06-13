import numpy as np
from unittest import TestCase, main
from denspp.offline.analog.pyspice_handler import PySpiceHandler


class TestPySpiceHandler(TestCase):
    voltage_mea = np.linspace(start=-5.0, stop=+5.0, num=101, endpoint=True, dtype=float)
    current_mea = voltage_mea / 100e3


if __name__ == '__main__':
    main()
