import numpy as np
from unittest import TestCase, main
from copy import deepcopy
from denspp.offline.metric import compare_timestamps
from .frame_generator_test import (
    _build_spike_signal,
    _build_sorted_timestamps
)
from .sda import (
    SettingsSDA,
    SpikeDetection
)


TestSettings = SettingsSDA(
    mode_align='min',
    mode_sda='normal',
    mode_thr='const',
    dx_sda=[1],
    sampling_rate=20e3,
    t_frame_length=1.6e-3,
    t_frame_start=0.4e-3,
    dt_offset=0.16e-3,
    thr_gain=1.0
)


class TestSettingsSDA(TestCase):
    def setUp(self):
        self.set0: SettingsSDA = deepcopy(TestSettings)

    def test_offset_integer(self):
        data_input = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset in data_input:
            self.set0.dt_offset = offset
            result.append(self.set0.get_integer_offset)

        check = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result, check)

    def test_integer_total(self):
        set0 = deepcopy(TestSettings)
        data_input0 = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset0 in data_input0:
            set0.dt_offset = offset0
            result.append(set0.get_integer_offset_total)

        check = [0, 4, 8, 12, 16, 20]
        np.testing.assert_array_equal(result, check)

    def test_integer_spike_size(self):
        set0 = deepcopy(TestSettings)
        data_input = [1.0e-3, 1.2e-3, 1.4e-3, 1.6e-3, 1.8e-3]

        result = list()
        for offset in data_input:
            set0.t_frame_length = offset
            result.append(set0.get_integer_spike_frame)

        check = [20, 24, 28, 32, 36]
        np.testing.assert_array_equal(result, check)

    def test_integer_spike_start(self):
        set0 = deepcopy(TestSettings)
        data_input = [0.0e-3, 0.1e-3, 0.2e-3, 0.3e-3, 0.4e-3, 0.5e-3]

        result = list()
        for offset in data_input:
            set0.t_frame_start = offset
            result.append(set0.get_integer_spike_start)

        check = [0, 2, 4, 6, 8, 10]
        np.testing.assert_array_equal(result, check)


class TestSpikeDetection(TestCase):
    def setUp(self):
        self.set0 = deepcopy(TestSettings)
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

    def test_methods_overview_sda(self):
        rslt = SpikeDetection(settings=self.set0).get_methods_sda()
        self.assertEqual(len(rslt), 7)
        self.assertTrue('normal' in rslt)
        self.assertTrue('spb' in rslt)
        self.assertTrue('neo' in rslt)

    def test_sda_none(self):
        self.set0.mode_sda = 'none'
        try:
            SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_sda_normal(self):
        self.set0.mode_sda = 'normal'
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        np.testing.assert_array_almost_equal(rslt, self.signal_eap[0])

    def test_sda_neo_ones(self):
        self.set0.mode_sda = 'neo'
        self.set0.dx_sda = [1]
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2.5e-9)

    def test_sda_neo_two(self):
        self.set0.mode_sda = 'neo'
        self.set0.dx_sda = [2]
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_sda_mteo_two(self):
        self.set0.mode_sda = 'mteo'
        self.set0.dx_sda = [1, 2, 3]
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_sda_ado_ones(self):
        self.set0.mode_sda = 'ado'
        self.set0.dx_sda = [1]
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_sda_ado_threes(self):
        self.set0.mode_sda = 'ado'
        self.set0.dx_sda = [3]
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_sda_eed(self):
        self.set0.mode_sda = 'eed'
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0], f_hp=150.0)
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_sda_spb(self):
        self.set0.mode_sda = 'spb'
        rslt = SpikeDetection(settings=self.set0).apply_spike_detection(xraw=self.signal_eap[0], f_bp=[100.0, 1000.0])
        self.assertEqual(rslt.size, self.signal_eap[0].size)
        self.assertGreater(rslt.min(), -2e-9)

    def test_spike_position_none(self):
        self.set0.mode_align = 'none'
        rslt = SpikeDetection(self.set0).get_spike_waveforms_from_positions(
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
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreater(rslt_pos.f1_score, .99)

    def test_spike_position_min(self):
        self.set0.mode_align = 'min'
        rslt = SpikeDetection(self.set0).get_spike_waveforms_from_positions(
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
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.get_integer_spike_start for _ in range(rslt.waveform.shape[0])])

    def test_spike_transient_normal_const(self):
        self.set0.mode_sda = 'normal'
        self.set0.mode_thr = 'const'
        self.set0.mode_align = 'none'
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False,
            thr_val=-40e-6
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=12
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreater(rslt_pos.f1_score, .94)

    def test_spike_transient_min_const(self):
        self.set0.mode_sda = 'normal'
        self.set0.mode_thr = 'const'
        self.set0.mode_align = 'min'
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False,
            thr_val=-40e-6
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=26
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreater(rslt_pos.f1_score, .94)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.get_integer_spike_start for _ in range(rslt.waveform.shape[0])])

    def test_spike_transient_neo(self):
        self.set0.dx_sda = [2]
        self.set0.mode_sda = 'neo'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 16)
        self.assertEqual(np.argmin(rslt.waveform, axis=1).tolist(),
                         [self.set0.get_integer_spike_start for _ in range(rslt.waveform.shape[0])])

    def test_spike_transient_mteo(self):
        self.set0.dx_sda = [2, 4, 6]
        self.set0.mode_sda = 'mteo'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 16)

    def test_spike_transient_ado(self):
        self.set0.dx_sda = [2]
        self.set0.mode_sda = 'ado'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 16)

    def test_spike_transient_aso(self):
        self.set0.dx_sda = [2]
        self.set0.mode_sda = 'aso'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 16)

    def test_spike_transient_eed(self):
        self.set0.dx_sda = [2]
        self.set0.mode_sda = 'eed'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False,
            f_hp=200.0
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 16)

    def test_spike_transient_spb(self):
        self.set0.dx_sda = [2]
        self.set0.mode_sda = 'spb'
        self.set0.mode_thr = 'rms_black'
        self.set0.mode_align = 'min'
        self.set0.thr_gain = 1
        rslt = SpikeDetection(self.set0).get_spike_waveforms(
            xraw=self.signal_eap[0],
            do_abs=False,
            f_bp=[200.0, 2000.0]
        )
        rslt_pos = compare_timestamps(
            true_labels=self.signal_eap[1].tolist(),
            pred_labels=rslt.xpos.tolist(),
            window=20
        )
        self.assertEqual(rslt.waveform.shape[1], self.set0.get_integer_spike_frame)
        self.assertGreaterEqual(rslt_pos.TP, 12)


if __name__ == '__main__':
    main()
