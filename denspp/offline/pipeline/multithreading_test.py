import unittest
import numpy as np
from .multithreading import ProcessingThread


def test_func(data: np.ndarray) -> dict:
    return {"data": data, "feat": np.mean(data, axis=0)}

class MultiThreadingTest(unittest.TestCase):
    num_channels = 5
    data = 5*(np.random.random((num_channels, 100))-0.5)
    chnl = np.linspace(start=0, stop=num_channels, num=num_channels, endpoint=False, dtype=int).tolist()

    def test_single_core(self):
        dut = ProcessingThread(
            num_workers=1
        )
        dut.do_processing(
            func=test_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt['0'].keys()] == ['data', 'feat'])

    def test_dual_core(self):
        dut = ProcessingThread(
            num_workers=2
        )
        dut.do_processing(
            func=test_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt['0'].keys()] == ['data', 'feat'])

    def test_quad_core(self):
        dut = ProcessingThread(
            num_workers=4
        )
        dut.do_processing(
            func=test_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt['0'].keys()] == ['data', 'feat'])

    def test_octo_core(self):
        dut = ProcessingThread(
            num_workers=8
        )
        dut.do_processing(
            func=test_func,
            data=self.data,
            chnnl_id=self.chnl
        )
        rslt = dut.get_results()
        self.assertTrue(len(rslt) == len(self.chnl) and [key for key in rslt['0'].keys()] == ['data', 'feat'])


if __name__ == '__main__':
    unittest.main()
