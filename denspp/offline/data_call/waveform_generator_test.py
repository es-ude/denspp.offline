import numpy as np
from unittest import TestCase, main
from denspp.offline.data_call import WaveformGenerator


# --- Info: Function have to start with test_*
class TestWaveformGenerator(TestCase):
    _sampling_rate: float = 20e3
    _period: float = 0.01

    def test_result_value_length(self):
        dict = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).get_dictionary_classes()
        self.assertTrue(len(dict) > 0)

    def test_result_value_available_waveforms(self):
        dict = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).get_dictionary_classes()
        types_to_check = ['RECT_HALF', 'LIN_RISE', 'LIN_FALL', 'SAW_POS', 'SAW_NEG']
        types_checked = [True for type in types_to_check if type in dict]
        self.assertTrue(np.sum(types_checked) == len(types_to_check))

    def test_waveform_zero_zero_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1/self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['ZERO'],
            polarity_cathodic=[False]
        ).signal
        length_zero = np.argwhere(signal == 0.).flatten()
        self.assertTrue(length_zero.size == 2+self._period*self._sampling_rate)

    def test_waveform_rect_zero_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1/self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['RECT_HALF'],
            polarity_cathodic=[False]
        ).signal
        length_zero = np.argwhere(signal == 0.).flatten()
        self.assertTrue(length_zero.size == 2)

    def test_waveform_rect_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['RECT_HALF'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(self._period * self._sampling_rate))

    def test_waveform_square_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['RECT_FULL'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(self._period * self._sampling_rate))

    def test_waveform_square_content(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[10 / self._sampling_rate],
            waveform_select=['RECT_FULL'],
            polarity_cathodic=[False]
        ).signal
        ref = np.array([ 0., 1., 1., 1., 1., 1., -1., -1., -1., -1., -1., 0.])
        np.testing.assert_almost_equal(signal, ref, decimal=8)

    def test_waveform_lin_rise_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['LIN_RISE'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_lin_fall_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['LIN_FALL'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_lin_rise_fall_equal(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate, 1 / self._sampling_rate, 1 / self._sampling_rate],
            time_duration=[self._period, self._period, self._period],
            waveform_select=['LIN_RISE', 'LIN_FALL', 'RECT_HALF'],
            polarity_cathodic=[False, False, True]
        ).signal
        np.testing.assert_almost_equal(signal, np.zeros_like(signal), decimal=8)

    def test_waveform_sine_half_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['SINE_HALF'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_sine_half_inv_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['SINE_HALF_INV'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_sine_half_equal(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate, 1 / self._sampling_rate, 1 / self._sampling_rate],
            time_duration=[self._period, self._period, self._period],
            waveform_select=['SINE_HALF', 'SINE_HALF_INV', 'RECT_HALF'],
            polarity_cathodic=[False, False, True]
        ).signal
        np.testing.assert_almost_equal(signal, np.zeros_like(signal), decimal=8)

    def test_waveform_sine_full_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['SINE_FULL'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_tri_half_content_length_one(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['TRI_HALF'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(self._period * self._sampling_rate))

    def test_waveform_tri_half_content_length_two(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[2 * self._period],
            waveform_select=['TRI_HALF'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 1, int(2 * self._period * self._sampling_rate))

    def test_waveform_tri_full_content_length_one(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['TRI_FULL'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 2, int(self._period * self._sampling_rate))

    def test_waveform_tri_full_content(self):
        signal = WaveformGenerator(12, add_noise=False).generate_waveform(
            time_points=[0.],
            time_duration=[1.],
            waveform_select=['TRI_FULL'],
            polarity_cathodic=[False]
        ).signal
        ref = np.array([ 0.,  0.33333333,  0.66666667,  1.,  0.66666667, 0.33333333, 0., -0.33333333, -0.66666667, -1., -0.66666667, -0.33333333])
        np.testing.assert_almost_equal(signal, ref, decimal=8)

    def test_waveform_saw_pos_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['SAW_POS'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(self._period * self._sampling_rate))

    def test_waveform_saw_neg_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['SAW_NEG'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(self._period * self._sampling_rate))

    def test_waveform_saw_pos_neg_equal(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate, 1 / self._sampling_rate],
            time_duration=[self._period, self._period],
            waveform_select=['SAW_POS', 'SAW_NEG'],
            polarity_cathodic=[False, False]
        ).signal
        np.testing.assert_almost_equal(signal, np.zeros_like(signal), decimal=8)

    def test_waveform_gauss_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['GAUSS'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(self._period * self._sampling_rate))

    def test_waveform_eap_content(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[2e-3],
            waveform_select=['EAP'],
            polarity_cathodic=[False]
        ).signal
        chck = np.array([0.0, -0.0012172548264579614, -0.005311060210664175, -0.019272136018855357, -0.05834180120914487, -0.14753128425548376, -0.31173514070946545, -0.5501866513568735, -0.8099373931559386, -0.9911547362864012, -1.0, -0.8135991915652375, -0.49668364555662703, -0.15395302356345117, 0.13388328120569107, 0.33706300246139936, 0.4618628339252224, 0.5245140250682183, 0.5376569993144354, 0.5102975228307609, 0.4518869951221461, 0.3741391312238784, 0.2897709373526756, 0.20996297854908397, 0.14233312007040994, 0.09027055407061382, 0.0535628125910162, 0.02973430540680821, 0.015442918658894482, 0.0075037461317305825, 0.003411175340723643, 0.0014507985640440527, 0.000577281118637366, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_almost_equal(signal, chck, decimal=4)

    def test_waveform_eap_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_waveform(
            time_points=[1 / self._sampling_rate],
            time_duration=[self._period],
            waveform_select=['EAP'],
            polarity_cathodic=[False]
        ).signal
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size, int(1.6e-3 * self._sampling_rate))

    def test_waveform_biphasic_content_length(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_biphasic_waveform(
            anodic_wvf='SINE_HALF',
            anodic_duration=self._period / 2,
            cathodic_wvf='SINE_HALF',
            cathodic_duration=self._period / 2,
            intermediate_duration=0.0,
            do_cathodic_first=False,
            do_charge_balancing=False
        )['y']
        length_content = np.argwhere(signal != 0.).flatten()
        self.assertEqual(length_content.size + 2, int(self._period * self._sampling_rate))

    def test_waveform_biphasic_charge_density(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_biphasic_waveform(
            anodic_wvf='SINE_HALF',
            anodic_duration=self._period / 2,
            cathodic_wvf='SINE_HALF',
            cathodic_duration=self._period / 2,
            intermediate_duration=0.0,
            do_cathodic_first=False,
            do_charge_balancing=False
        )['y']
        dq = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).check_charge_balancing(signal)
        self.assertEqual(dq, 0.0)

    def test_waveform_biphasic_asymmetric_charge_density(self):
        signal = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).generate_biphasic_waveform(
            anodic_wvf='SINE_HALF',
            anodic_duration=self._period/2,
            cathodic_wvf='SINE_HALF',
            cathodic_duration=self._period,
            intermediate_duration=0.0,
            do_cathodic_first=False,
            do_charge_balancing=True
        )['y']
        dq = WaveformGenerator(sampling_rate=self._sampling_rate, add_noise=False).check_charge_balancing(signal)
        np.testing.assert_almost_equal(dq, 0.0, decimal=2)

    def test_waveform_quant_sine_unsigned_unoptimized(self):
        out = WaveformGenerator(sampling_rate=12, add_noise=False).generate_waveform_quant_fxp(
            time_points=[0],
            time_duration=[1],
            waveform_select=['SINE_FULL'],
            polarity_cathodic=[False],
            bitwidth=6,
            bitfrac=0,
            signed=False,
            do_opt=False
        ).signal
        ref = np.array([32, 48, 59, 63, 59, 48, 32, 16, 4, 0, 4, 15], dtype=np.int32)
        np.testing.assert_almost_equal(out, ref, decimal=4)

    def test_waveform_quant_sine_signed_unoptimized(self):
        out = WaveformGenerator(sampling_rate=12, add_noise=False).generate_waveform_quant_fxp(
            time_points=[0],
            time_duration=[1],
            waveform_select=['SINE_FULL'],
            polarity_cathodic=[False],
            bitwidth=6,
            bitfrac=0,
            signed=True,
            do_opt=False
        ).signal
        ref = np.array([0, 15, 27, 31, 27, 16, 0, -15, -27, -32, -27, -16], dtype=np.int32)
        np.testing.assert_almost_equal(out, ref, decimal=4)

    def test_waveform_quant_sine_signed_optimized(self):
        out = WaveformGenerator(sampling_rate=12, add_noise=False).generate_waveform_quant_fxp(
            time_points=[0],
            time_duration=[1],
            waveform_select=['SINE_FULL'],
            polarity_cathodic=[False],
            bitwidth=6,
            bitfrac=0,
            signed=True,
            do_opt=True
        ).signal
        ref = np.array([  0,  15,  27,  31], dtype=np.int32)
        np.testing.assert_almost_equal(out, ref, decimal=4)

    def test_waveform_quant_triangular_unsigned_unoptimized(self):
        out = WaveformGenerator(sampling_rate=12, add_noise=False).generate_waveform_quant_fxp(
            time_points=[0],
            time_duration=[1],
            waveform_select=['TRI_FULL'],
            polarity_cathodic=[False],
            bitwidth=6,
            bitfrac=0,
            signed=False,
            do_opt=False
        ).signal
        ref = np.array([32, 42, 53, 63, 53, 42, 32, 21, 10, 0, 10, 21], dtype=np.int32)
        np.testing.assert_almost_equal(out, ref, decimal=4)

    def test_build_random_timestamps(self):
        rslt = WaveformGenerator(sampling_rate=20e3, add_noise=False).build_random_timestamps(
            count=100,
            min_gap=0.002,
            max_gap=0.01
        )
        self.assertEqual(len(rslt), 100)


if __name__ == '__main__':
    main()
