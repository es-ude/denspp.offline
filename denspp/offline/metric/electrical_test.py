import unittest

import numpy as np

from denspp.offline.metric.electrical import (
    calculate_cosine_similarity,
    calculate_total_harmonics_distortion,
    calculate_total_harmonics_distortion_from_transient,
)
from denspp.offline.preprocessing import do_fft


class TestElectricalMetric(unittest.TestCase):
    def test_metric_thd_one_harmonic_tran(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = (
            np.sin(2 * np.pi * 50 * t)
            + 0.1 * np.sin(2 * np.pi * 100 * t)
            + 0.05 * np.sin(2 * np.pi * 150 * t)
        )

        rslt = calculate_total_harmonics_distortion_from_transient(
            signal=signal, fs=sampling_rate, N_harmonics=1
        )
        self.assertAlmostEqual(rslt, -20.067970271376048, delta=0.1)

    def test_metric_thd_one_harmonic_spec(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = (
            np.sin(2 * np.pi * 50 * t)
            + 0.1 * np.sin(2 * np.pi * 100 * t)
            + 0.05 * np.sin(2 * np.pi * 150 * t)
        )

        freq, spec = do_fft(
            y=signal,
            fs=sampling_rate,
        )

        rslt = calculate_total_harmonics_distortion(freq=freq, spectral=spec, N_harmonics=2)
        self.assertAlmostEqual(rslt, -19.11810872201894, delta=0.1)

    def test_calculate_cosine_match(self):
        t = np.linspace(0, 1, 1000, endpoint=True)
        rslt = calculate_cosine_similarity(
            y_pred=np.sin(2 * np.pi * 50 * t),
            y_true=np.sin(2 * np.pi * 50 * t),
        )
        self.assertEqual(rslt, 1.0)

    def test_calculate_cosine_half(self):
        t = np.linspace(0, 1, 1000, endpoint=True)
        rslt = calculate_cosine_similarity(
            y_pred=np.sin(2 * np.pi * 50 * t),
            y_true=np.sin(2 * np.pi * 100 * t),
        )
        self.assertAlmostEqual(rslt, -4.3151246464923076e-17, delta=1e-12)


if __name__ == "__main__":
    unittest.main()
