import unittest
from math import isclose
import numpy as np
import torch
from .snr import (
    calculate_snr,
    calculate_snr_tensor,
    calculate_dsnr_tensor,
    calculate_snr_cluster
)


def _build_noise_numpy(scale: float, size: int) -> np.ndarray:
    return scale * 4 * (np.random.rand(size) - 0.5)


def _build_noise_tensor(scale: float, shape: torch.Size) -> torch.Tensor:
    return scale * 4 * (torch.rand(shape) - 0.5)


class TestMetricSNR(unittest.TestCase):
    def setUp(self):
        fs_ana = 1e3
        t_end = 10*np.pi
        time = np.linspace(start=0., stop=t_end, num=int(t_end * fs_ana), endpoint=True)
        scale_amp = 1.
        self.signal_numpy = scale_amp * np.sin(time * 10.)
        self.signal_torch = scale_amp * torch.sin(torch.Tensor(time) * 10.).unsqueeze(0)
        self.signal_torch = self.signal_torch.repeat(10, 1)

    def test_snr_numpy_plus_60db(self):
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.00001, self.signal_numpy.size)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, 60., abs_tol=0.5))

    def test_snr_numpy_minus_0db(self):
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.01, self.signal_numpy.size)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, 0., abs_tol=0.5))

    def test_snr_numpy_minus_20db(self):
        snr_out = list()
        for _ in range(10):
            noise = _build_noise_numpy(0.1, self.signal_numpy.size)
            signal = noise + self.signal_numpy
            rslt = calculate_snr(
                data=signal,
                mean=self.signal_numpy,
            )
            snr_out.append(rslt)
        rslt = np.mean(np.array(snr_out))
        self.assertTrue(isclose(rslt, -20., abs_tol=0.5))

    def test_snr_tensor_plus_60db(self):
        noise = _build_noise_tensor(0.00001, self.signal_torch.shape)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(
            data=signal,
            mean=self.signal_torch
        )
        rslt = np.mean(np.array(rslt))
        self.assertTrue(isclose(rslt, 60., abs_tol=0.5))

    def test_snr_tensor_minus_0db(self):
        noise = _build_noise_tensor(0.01, self.signal_torch.shape)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(
            data=signal,
            mean=self.signal_torch
        )
        rslt = np.mean(np.array(rslt))
        self.assertTrue(isclose(rslt, 0., abs_tol=0.5))

    def test_snr_tensor_minus_20db(self):
        noise = _build_noise_tensor(0.1, self.signal_torch.shape)
        signal = noise + self.signal_torch
        rslt = calculate_snr_tensor(
            data=signal,
            mean=self.signal_torch
        )
        rslt = np.mean(np.array(rslt))
        self.assertTrue(isclose(rslt, -20., abs_tol=0.5))

    def test_dsnr_tensor(self):
        noise0 = _build_noise_tensor(0.1, self.signal_torch.shape)
        noise1 = _build_noise_tensor(0.01, self.signal_torch.shape)
        signal0 = noise0 + self.signal_torch
        signal1 = noise1 + self.signal_torch
        rslt = calculate_dsnr_tensor(
            data=signal0,
            pred=signal1,
            mean=self.signal_torch
        )
        rslt = np.mean(np.array(rslt))
        self.assertTrue(isclose(rslt, +20., abs_tol=0.5))

    def test_snr_cluster(self):
        rslt = calculate_snr_cluster()


if __name__ == '__main__':
    unittest.main()
