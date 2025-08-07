import unittest
import numpy as np
from denspp.offline.preprocessing.transformation import do_fft
from denspp.offline.metric.electrical import (calculate_total_harmonics_distortion,
                                              calculate_total_harmonics_distortion_from_transient,
                                              calculate_cosine_similarity)


class TestElectricalMetric(unittest.TestCase):
    def test_metric_thd_one_harmonic_tran(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        rslt = calculate_total_harmonics_distortion_from_transient(
            signal=signal,
            fs=sampling_rate,
            N_harmonics=1
        )
        self.assertEqual(rslt, -20.067970271376048)

    def test_metric_thd_one_harmonic_spec(self):
        sampling_rate = 1000
        t = np.linspace(0, 1, sampling_rate, endpoint=True)
        signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.sin(2 * np.pi * 100 * t) + 0.05 * np.sin(2 * np.pi * 150 * t)

        freq, spec = do_fft(
            y=signal,
            fs=sampling_rate,
        )

        rslt = calculate_total_harmonics_distortion(
            freq=freq,
            spectral=spec,
            N_harmonics=2
        )
        self.assertEqual(rslt, -19.118108722018935)

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
        self.assertEqual(rslt, -4.70001641772466e-17)


if __name__ == "__main__":
    unittest.main()
