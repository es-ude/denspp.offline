import unittest
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from denspp.offline.data_call import WaveformGenerator
from denspp.offline.metric import compare_timestamps
from .frame_generator import FrameWaveform, DefaultSettingsFrame, SettingsFrame, FrameGenerator


def _build_spike_waveform(sampling_rate: float, do_plot: bool=False) -> np.ndarray:
    eap = WaveformGenerator(
        sampling_rate=sampling_rate,
        add_noise=False
    ).generate_waveform(
        time_points=[0.],
        time_duration=[1.6e-3],
        waveform_select=['EAP'],
        polarity_cathodic=[False]
    )

    if do_plot:
        plt.plot(eap['time'], eap['sig'], 'k', marker='.')
        plt.grid(True)
        plt.show()

    return eap['sig']


def _build_spike_signal(scale_pp_range: list[float], scale_noise: float, pos_spike: list[float], sampling_rate: float, do_noise: bool=True, do_plot: bool=False) -> tuple[np.ndarray, np.ndarray]:
    t_end_sim = float(pos_spike[-1] + 3.2e-3)
    time = np.linspace(start=0., stop=t_end_sim, num=int(t_end_sim * sampling_rate), endpoint=True)
    spike_signal = np.zeros_like(time)
    spike_pos = np.zeros_like(pos_spike, dtype=np.int32)
    spike_template = _build_spike_waveform(sampling_rate, False)
    for idx, pos in enumerate(pos_spike):
        pos_start = int(pos * sampling_rate)
        pos_end = pos_start + spike_template.size
        scale_pp = (scale_pp_range[1] - scale_pp_range[0]) * np.random.rand(1) + scale_pp_range[0]

        spike_signal[pos_start:pos_end] = scale_pp * spike_template
        spike_pos[idx] = pos_start + int(spike_template.size /2)

    spike_signal += np.zeros_like(spike_signal) if not do_noise else 0.5 * scale_noise * np.random.randn(spike_signal.size)

    if do_plot:
        plt.plot(time, spike_signal, 'k', marker='.')
        plt.plot(time[spike_pos.tolist()], spike_signal[spike_pos.tolist()], 'r', marker='.', linestyle='None')
        plt.grid(True)
        plt.show()
    return spike_signal, spike_pos


def _build_sorted_timestamps(count: int, min_gap: float=0.002, max_gap: float=0.01) -> list:
    return WaveformGenerator(
        sampling_rate=0.,
        add_noise=False
    ).build_random_timestamps(
        count=count,
        min_gap=min_gap,
        max_gap=max_gap,
    )


def _plot_frame_waveforms(frames: FrameWaveform, do_plot: bool=False) -> None:
    if do_plot:
        plt.figure()
        plt.plot(np.transpose(frames.waveform))
        plt.xlim([0, frames.length-1])
        plt.grid()
        plt.show()
    else:
        pass


class TestBuildEAP(unittest.TestCase):
    def test_build_spike_waveform(self):
        signal = _build_spike_waveform(
            sampling_rate=20e3,
            do_plot=False
        )
        assert signal.size == 32
        self.assertEqual(signal.min(), -1.0)

    def test_build_spike_signal_specific(self):
        signal = _build_spike_signal(
            scale_pp_range=[80e-6, 120e-6],
            scale_noise=10e-6,
            pos_spike=[0.01, 0.21, 0.3, 0.34, 0.42, 0.44, 0.46, 0.48, 0.8, 1.],
            sampling_rate=20e3,
            do_plot=False
        )
        self.assertEqual(signal[1].size, 10)

    def test_build_spike_signal_random(self):
        signal = _build_spike_signal(
            scale_pp_range=[80e-6, 120e-6],
            scale_noise=10e-6,
            pos_spike=_build_sorted_timestamps(
                count=100,
                min_gap=0.005,
                max_gap=0.1,
            ),
            sampling_rate=20e3,
            do_plot=False
        )
        self.assertEqual(signal[1].size, 100)

    def test_build_sorted_timestamps(self):
        pos = _build_sorted_timestamps(
            count=10,
            min_gap=2e-3,
            max_gap=20e-3
        )
        self.assertEqual(len(pos), 10)


class TestSettingsFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFrame = deepcopy(DefaultSettingsFrame)

    def test_integer_waveform_length(self):
        self.set0.sampling_rate = 20e3
        input = [1.6e-3, 2e-3, 2.4e-3]
        chck = [int(val * self.set0.sampling_rate) for val in input]

        for test_val, true_val in zip(input, chck):
            self.set0.window_sec = test_val
            rslt = self.set0.length_frame_int
            self.assertEqual(rslt, true_val)

    def test_integer_offset_length(self):
        self.set0.sampling_rate = 20e3
        input = [1.6e-3, 2e-3, 2.4e-3]
        chck = [int(val * self.set0.sampling_rate) for val in input]

        for test_val, true_val in zip(input, chck):
            self.set0.offset_sec = test_val
            rslt = self.set0.length_offset_int
            self.assertEqual(rslt, true_val)

    def test_integer_align_position(self):
        self.set0.sampling_rate = 20e3
        input = [1.6e-3, 2e-3, 2.4e-3]
        chck = [int(val * self.set0.sampling_rate) for val in input]

        for test_val, true_val in zip(input, chck):
            self.set0.align_sec = test_val
            rslt = self.set0.length_align_position
            self.assertEqual(rslt, true_val)

    def test_integer_total_frame_int(self):
        self.set0.sampling_rate = 20e3
        self.set0.window_sec = 2e-3
        self.set0.offset_sec = 0.2e-3

        rslt = self.set0.length_total_frame
        chck = int(self.set0.sampling_rate * (2* self.set0.offset_sec + self.set0.window_sec))
        self.assertEqual(rslt, chck)


class TestFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.set0: SettingsFrame = deepcopy(DefaultSettingsFrame)
        self.set0.window_sec = 1.6e-3
        self.signal_eap = _build_spike_signal(
            scale_pp_range=[80e-6, 120e-6],
            scale_noise=10e-6,
            pos_spike=_build_sorted_timestamps(
                count=20,
                min_gap=2e-3,
                max_gap=20e-3
            ),
            sampling_rate=self.set0.sampling_rate,
            do_noise=True,
            do_plot=False
        )
        self.frames_eap = np.array([_build_spike_waveform(
            sampling_rate=self.set0.sampling_rate,
            do_plot=False
        ) for _ in range(10)])

    def test_methods_overview_frame_aligning(self):
        rslt = FrameGenerator(self.set0).get_methods_frame_aligning()
        self.assertEqual(len(rslt), 6)
        self.assertTrue('none' in rslt)
        self.assertTrue('max' in rslt)

    def test_get_threshold_with_constant(self):
        self.set0.mode_thr = 'const'
        rslt = FrameGenerator(self.set0).get_threshold(self.signal_eap[0], thr_val=-60e-6)
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        np.testing.assert_array_equal(rslt, np.zeros_like(self.signal_eap[0]) - 60e-6)

    def test_get_threshold_position_with_constant(self):
        self.set0.mode_thr = 'const'
        rslt = FrameGenerator(self.set0).get_threshold_position(self.signal_eap[0], thr_val=-60e-6)
        self.assertEqual(rslt.size, self.signal_eap[1].size)

    def test_get_align_frames_none(self):
        self.set0.mode_align = 'none'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_get_align_frames_max(self):
        self.set0.mode_align = 'max'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_get_align_frames_min(self):
        self.set0.mode_align = 'min'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_get_align_frames_ntp(self):
        self.set0.mode_align = 'ntp'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_get_align_frames_ptp(self):
        self.set0.mode_align = 'ptp'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_get_align_frames_absmax(self):
        self.set0.mode_align = 'absmax'
        FrameGenerator(self.set0).get_aligning_position(
            frame_in=self.frames_eap[0]
        )
        self.assertTrue(True)

    def test_frame_generation_position_none(self):
        self.set0.mode_align = 'none'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-20
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        _plot_frame_waveforms(rslt)

    def test_frame_generation_position_min(self):
        self.set0.mode_align = 'min'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-18
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=16
        )
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])
        _plot_frame_waveforms(rslt)

    def test_frame_generation_position_max(self):
        self.set0.mode_align = 'max'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-18
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=16
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        self.assertEqual(np.argmax(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_position_absmax(self):
        self.set0.mode_align = 'absmax'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-18
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=16
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        self.assertEqual(np.argmax(np.abs(rslt.waveform), axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_position_ntp(self):
        self.set0.mode_align = 'ntp'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-18
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=25
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        self.assertEqual(np.argmin(np.diff(rslt.waveform), axis=1).tolist(),
                         [self.set0.length_align_position-1 for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_position_ptp(self):
        self.set0.mode_align = 'ptp'
        rslt = FrameGenerator(self.set0).frame_generation_with_position(
            xraw=self.signal_eap[0],
            xpos=self.signal_eap[1],
            xoffset=-18
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=16
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[0], self.signal_eap[1].size)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .99)
        self.assertEqual(np.argmax(np.diff(rslt.waveform), axis=1).tolist(),
                         [self.set0.length_align_position-1 for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_normal_const(self):
        self.set0.mode_thr = 'const'
        self.set0.mode_align = 'none'
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False,
            thr_val=-40e-6
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=12
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        _plot_frame_waveforms(rslt)

    def test_frame_generation_transient_const(self):
        self.set0.mode_thr = 'const'
        self.set0.mode_align = 'min'
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False,
            thr_val=-40e-6
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])
        _plot_frame_waveforms(rslt)

    def test_frame_generation_transient_absmean(self):
        self.set0.mode_thr = 'abs_mean'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = -5.2
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_transient_moving_average(self):
        self.set0.mode_thr = 'mavg_abs'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = -4.5
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreaterEqual(rslt_pos.f1_score, .9)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_transient_rms_norm(self):
        self.set0.mode_thr = 'rms_norm'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = -4.5
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_transient_rms_blackrock(self):
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = -1.
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])

    def test_frame_generation_transient_welford(self):
        self.set0.mode_thr = 'welford'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = -4.5
        rslt = FrameGenerator(self.set0).frame_generation(
            xraw=self.signal_eap[0],
            xsda=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        _plot_frame_waveforms(rslt)
        self.assertEqual(rslt.waveform.shape[1], self.set0.length_frame_int)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.length_align_position for _ in range(rslt.waveform.shape[0])])


if __name__ == '__main__':
    unittest.main()
