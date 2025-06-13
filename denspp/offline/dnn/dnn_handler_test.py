import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.dnn.dnn_handler import preprocessing_dnn


# --- Info: Function have to start with test_*
class TestDNNHandler(TestCase):
    def test_preparing_func(self):
        rslt = preprocessing_dnn()
        self.assertTrue(True)


if __name__ == '__main__':
    main()
