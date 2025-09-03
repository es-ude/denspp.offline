import unittest
import numpy as np
from .multicore import MultiprocessingHandler


def dummy_func(data: np.ndarray) -> dict:
    return {"data": data, "feat": np.mean(data, axis=0)}


class MultiThreadingTest(unittest.TestCase):
    num_channels = 16
    data = np.random.random((num_channels, 100))-0.5
    chnl = np.linspace(start=0, stop=num_channels, num=num_channels, endpoint=False, dtype=int).tolist()

    def test_single_core(self):
        dut = MultiprocessingHandler(
            num_workers=1
        )
        dut.do_processing(
            func=dummy_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        first_key = [key for key in rslt.keys()][0]
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt[first_key].keys()] == ['data', 'feat'])

    @unittest.skip
    def test_dual_core(self):
        dut = MultiprocessingHandler(
            num_workers=2
        )
        dut.do_processing(
            func=dummy_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        first_key = [key for key in rslt.keys()][0]
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt[first_key].keys()] == ['data', 'feat'])

    @unittest.skip
    def test_quad_core(self):
        dut = MultiprocessingHandler(
            num_workers=4
        )
        dut.do_processing(
            func=dummy_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        first_key = [key for key in rslt.keys()][0]
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt[first_key].keys()] == ['data', 'feat'])


if __name__ == '__main__':
    unittest.main()
