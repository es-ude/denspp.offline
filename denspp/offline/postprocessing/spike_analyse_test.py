import unittest
import numpy as np
import matplotlib.pyplot as plt
from denspp.offline.data_call import WaveformGenerator
from denspp.offline.preprocessing import FrameWaveform
from .spike_analyse import (
    calc_amplitude,
    calc_spike_ticks,
    calc_autocorrelogram,
    calc_firing_rate,
    calc_interval_timing
)


def _build_eap_dataset(count: int, sampling_rate: float, max_gap: float=0.01, do_plot: bool=False) -> FrameWaveform:
    dut = WaveformGenerator(
        sampling_rate=sampling_rate,
        add_noise=False
    )
    wvf = 100e-6 * dut.generate_waveform(
        time_points=[0.],
        time_duration=[1.6e-3],
        waveform_select=['EAP'],
        polarity_cathodic=[False]
    )['sig']
    pos = np.array(dut.build_random_timestamps(
        count=2* count,
        min_gap=1.8e-3,
        max_gap=max_gap,
    )) * sampling_rate
    wvf_noise = np.tile(wvf, (2*count, 1)) + 8e-6 * np.random.randn(2*count, wvf.size)
    wvf_noise[count:] = (-1) * wvf_noise[count:]
    label = np.zeros((2* count, ), dtype=int)
    label[count:] = label[count:] + 1

    out = FrameWaveform(
        waveform=wvf_noise,
        xpos=np.array(pos, dtype=int),
        label=label,
        sampling_rate=sampling_rate
    )
    if do_plot:
        plt.plot(np.transpose(out.waveform))
        plt.xlim([0, out.length-1])
        plt.grid()
        plt.show()
    return out


class SpikeAnalysisTest(unittest.TestCase):
    show_plots: bool = True

    def setUp(self):
        self.fs = 20e3
        self.dataset = _build_eap_dataset(
            count=12,
            sampling_rate=self.fs,
            max_gap=0.1,
            do_plot=False
        )

    def test_build_dataset(self):
        self.assertEqual(self.dataset.xpos.size, 24)
        self.assertEqual(self.dataset.waveform.shape, (24, 32))

    def test_calc_amplitude(self):
        rslt = calc_amplitude(
            frames_in=self.dataset
        )
        self.assertEqual(len(rslt), 2)
        self.assertEqual(len(rslt[0]), 3)
        self.assertEqual(len(rslt[1]), 3)
        self.assertEqual([key for key in rslt[0].keys()], ['pos', 'min', 'max'])
        self.assertEqual(len(rslt[1]['pos']), 12)

    def test_calc_spiketicks(self):
        rslt = calc_spike_ticks(
            spike=self.dataset
        )
        self.assertEqual(rslt.shape, (2, 24))
        self.assertEqual(rslt[1, 11], 0)
        self.assertEqual(rslt[1, 12], 1)

    def test_calc_autocorrelegram(self):
        ticks = calc_spike_ticks(spike=self.dataset)
        rslt = calc_autocorrelogram(
            ticks=ticks,
            fs=self.dataset.sampling_rate
        )
        self.assertEqual(len(rslt), 2)
        self.assertEqual(rslt[0].shape, (144, ))
        self.assertEqual(rslt[1].shape, (144, ))

    def test_calc_firing_rate(self):
        ticks = calc_spike_ticks(spike=self.dataset)
        rslt = calc_firing_rate(
            ticks=ticks,
            fs=self.fs
        )
        self.assertEqual(len(rslt), 2)
        self.assertEqual(rslt[0].shape, (2, 12))
        self.assertEqual(rslt[1].shape, (2, 12))
        self.assertEqual(rslt[1][1,:].min(), 0.0)
        self.assertGreater(rslt[1][1,:].max(), 0.0)
        self.assertLessEqual(rslt[1][1,:].max(), 300)

    def test_calc_interval_timing(self):
        ticks = calc_spike_ticks(spike=self.dataset)
        rslt = calc_interval_timing(
            ticks=ticks,
            fs=self.fs
        )
        self.assertEqual(len(rslt), 2)
        self.assertEqual(rslt[0].shape, (11, ))
        self.assertEqual(rslt[1].shape, (11, ))
        self.assertGreater(rslt[1].min(), 2e-3)


if __name__ == '__main__':
    unittest.main()
