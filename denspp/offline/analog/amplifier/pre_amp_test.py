import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.analog.amplifier.pre_amp import SettingsAMP, PreAmp


settings = SettingsAMP(
    vdd=0.6, vss=-0.6,
    fs_ana=50e3, gain=40,
    n_filt=1, f_filt=[0.1, 8e3], f_type="bandpass",
    offset=0e-6,
    f_chop=50,
    noise_en=False,
    noise_edev=100e-9
)


# --- Info: Function have to start with test_*
class TestAmplifier(TestCase):
    def test_settings_amp_vcm_bipolar(self):
        set0 = deepcopy(settings)
        set0.vdd = +0.6
        set0.vss = -0.6
        self.assertEqual(set0.vcm, 0.0)

    def test_settings_amp_vcm_uniolar(self):
        set0 = deepcopy(settings)
        set0.vdd = 0.6
        set0.vss = 0.0
        self.assertEqual(set0.vcm, 0.3)

    def test_amplifier_coeffs(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_type = 'low'
        set0.f_filt = [25]
        coeffs = PreAmp(settings_dev=set0).get_filter_coeffs
        check = np.sum(np.abs(coeffs['a'] - np.array([ 1.0, -5.551115123125783e-17])) + np.abs(coeffs['b'] - np.array([0.5, 0.5])))
        self.assertTrue(check <= 1e-6)

    def test_amplifier_preamp_behav(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_type = 'low'
        set0.f_filt = [25]
        set0.noise_en = False

        num_rpt = 2
        f_sig = 100 / set0.fs_ana
        input_time = np.linspace(start=-num_rpt / f_sig, stop=1 / f_sig, num=int((num_rpt + 1) * f_sig * set0.fs_ana),
                                 endpoint=True)
        input_stimuli = 0.5 * np.sin(np.pi * f_sig * input_time)
        output = PreAmp(settings_dev=set0).pre_amp(uinp=input_stimuli, uinn=0.0)
        check = True
        start_pos = np.argwhere(input_time >= 0.0).flatten()[0]
        diff_value = np.sum(np.abs(input_stimuli[start_pos:] - output[start_pos:])) / set0.gain
        self.assertTrue(diff_value <= 0.6)

    def test_amplifier_preamp_noise(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_type = 'low'
        set0.f_filt = [25]
        set0.noise_en = True
        set0.noise_edev = 100e-9

        input_noise = np.zeros(shape=(100,), dtype=float)
        out_noise = PreAmp(settings_dev=set0).pre_amp(uinp=input_noise, uinn=0.0)
        ref_check = 3 * set0.noise_edev * np.sqrt(set0.fs_ana)
        val_check = np.max(np.abs(out_noise))
        self.assertTrue(val_check <= ref_check and val_check > 0.0)

    def test_amplifier_chopper_behav(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_chop = set0.fs_ana / 10
        set0.f_type = 'low'
        set0.f_filt = [25]
        set0.noise_en = False

        num_rpt = 2
        f_sig = 100 / set0.fs_ana
        input_time = np.linspace(start=-num_rpt / f_sig, stop=1 / f_sig, num=int((num_rpt + 1) * f_sig * set0.fs_ana),
                                 endpoint=True)
        input_stimuli = 0.5 * np.sin(np.pi * f_sig * input_time)
        output = PreAmp(settings_dev=set0).pre_amp_chopper(uinp=input_stimuli, uinn=0.0)['out']
        start_pos = np.argwhere(input_time >= 0.0).flatten()[0]
        diff_value = np.sum(np.abs(input_stimuli[start_pos:] - output[start_pos:])) / set0.gain
        self.assertTrue(diff_value <= 0.6)

    def test_amplifier_chopper_bandpass(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_chop = set0.fs_ana / 10
        set0.f_type = 'bandpass'
        set0.f_filt = [25., 40.]
        set0.noise_en = False

    def test_amplifier_chopper_noise(self):
        set0 = deepcopy(settings)
        set0.n_filt = 1
        set0.gain = 1
        set0.fs_ana = 100
        set0.f_chop = set0.fs_ana / 10
        set0.f_type = 'low'
        set0.f_filt = [25]
        set0.noise_en = True
        set0.noise_edev = 100e-9

        input_noise = np.zeros(shape=(100, ), dtype=float)
        out_noise = PreAmp(settings_dev=set0).pre_amp_chopper(uinp=input_noise, uinn=0.0)['out']
        ref_check = 3 * set0.noise_edev * np.sqrt(set0.fs_ana)
        val_check = np.max(np.abs(out_noise))
        self.assertTrue(val_check <= ref_check and val_check > 0.0)


if __name__ == '__main__':
    main()
