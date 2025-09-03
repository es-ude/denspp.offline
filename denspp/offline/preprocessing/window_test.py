import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.preprocessing import SettingsWindow, WindowSequencer, transformation_window_method


class TestWindowMethod(TestCase):
    time = np.linspace(start=0, stop=100e-3, num=2000, endpoint=False, dtype=float)
    vsig = np.sin(2 * np.pi * 100. * time) + 0.25 * np.sin(2 * np.pi * 1000. * time)
    fs = float(1 / np.diff(time).min())
    
    def test_transformation_window_method_false(self):
        try:
            window = transformation_window_method(
                window_size=self.vsig.size,
                method = 'Hammingd'
            )
        except:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test_transformation_window_method_ones(self):
        window = transformation_window_method(
            window_size=self.vsig.size,
            method=''
        )
        np.testing.assert_array_equal(window, np.ones_like(self.vsig))


class TestSettingsWindowSequencer(TestCase):
    sets = SettingsWindow(
        sampling_rate=10e3,
        window_sec=10e-3,
        overlap_sec=0.1e-3
    )

    def test_settings_length(self):
        self.assertEqual(self.sets.window_length, 100)

    def test_settings_overlap(self):
        self.assertEqual(self.sets.overlap_length, 1)


class TestWindowSequencer(TestCase):
    sets = SettingsWindow(
        sampling_rate=10e3,
        window_sec=10e-3,
        overlap_sec=0.1e-3
    )

    def test_window_sequence_match_full(self):
        set0 = deepcopy(self.sets)
        num_trials = np.random.randint(1, 10)
        num_samples = int(num_trials * set0.window_length)
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).sequence(stimuli)

        assert rslt.shape == (int(num_trials), set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_sequence_match_not_full(self):
        set0 = deepcopy(self.sets)
        num_trials = np.random.randint(1, 10) + np.random.random(1) - 0.5
        num_samples = int(num_trials * set0.window_length)
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).sequence(stimuli)

        assert rslt.shape == (int(num_trials), set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_slide_max_overlapping(self):
        set0 = deepcopy(self.sets)
        set0.overlap_sec = set0.window_sec - 1 / set0.sampling_rate
        num_trials = np.random.randint(1, 10)
        num_samples = int(num_trials * set0.window_length)
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).slide(stimuli)

        assert rslt.shape == (int((num_trials - 1) * set0.window_length) + 1, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_slide_quarter_overlapping(self):
        set0 = deepcopy(self.sets)
        set0.overlap_sec = 0.25* set0.window_sec
        num_trials = np.random.randint(1, 10)
        num_samples = int(num_trials * set0.window_length)
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).slide(stimuli)

        assert rslt.shape[0] in range(int(num_trials*1.5-2), int(num_trials*1.5))
        assert rslt.shape[1] == set0.window_length
        for idx, sequence in enumerate(rslt):
            start_point = int(idx * self.sets.window_length * 0.75)
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_slide_half_overlapping(self):
        set0 = deepcopy(self.sets)
        set0.overlap_sec = 0.5* set0.window_sec
        num_trials = np.random.randint(1, 10)
        num_samples = int(num_trials * set0.window_length)
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).slide(stimuli)

        assert rslt.shape == (num_trials*2-1, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = int(idx * self.sets.window_length * 0.5
                              )
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_slide_sequence(self):
        set0 = deepcopy(self.sets)
        set0.overlap_sec = 0.
        num_trials = np.random.randint(1, 10)
        num_samples = num_trials * set0.window_length
        stimuli = 2 * (np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).slide(stimuli)

        assert rslt.shape == (num_trials, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)

    def test_window_event_detection_without_padding_normal(self):
        set0 = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0).window_event_detected(
            signal=stimuli,
            thr=0.25,
            pre_time=0.01,
            do_abs=False
        )
        self.assertEqual(sequence.shape, (num_trials, set0.window_length))
        chck0 = [np.sum(frame == frame[0]) for frame in sequence]
        self.assertTrue(all(x == chck0[0] for x in chck0))

    def test_window_event_detection_without_padding_absolute(self):
        set0 = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0).window_event_detected(
            signal=stimuli,
            thr=0.25,
            pre_time=0.01,
            do_abs=True
        )
        self.assertEqual(sequence.shape, (2 * num_trials, set0.window_length))
        chck0 = [np.sum(frame == frame[0]) for frame in sequence]
        self.assertTrue(all(x == chck0[0] for x in chck0))

    def test_window_event_detection_with_prepadding(self):
        set0 = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2 * np.pi * np.arange(start=0, stop=num_trials, step=1 / set0.sampling_rate))
        sequence = WindowSequencer(set0).window_event_detected(
            signal=stimuli,
            thr=0.25,
            pre_time=0.05
        )
        chck0 = np.sum(sequence[0, :] == sequence[0, 0])
        chck1 = np.sum(sequence[1, :] == sequence[1, 0])
        self.assertGreater(chck0, chck1)

    def test_window_event_detection_with_postpadding(self):
        set0 = deepcopy(self.sets)
        set0.window_sec = 0.25
        num_trials = 5
        stimuli = np.sin(2* np.pi* np.arange(start=0, stop=num_trials, step=1/set0.sampling_rate))
        stimuli[int(set0.window_length/2)] = 1.5
        stimuli[-int(set0.window_length/2)] = 1.5
        sequence = WindowSequencer(set0).window_event_detected(
            signal=stimuli,
            thr=1.2,
            pre_time=0.05,
            do_abs=False
        )
        chck0 = np.sum(sequence[-2, :] == sequence[-2, -1])
        chck1 = np.sum(sequence[-1, :] == sequence[-1, -1])
        self.assertGreater(chck1, chck0)


if __name__ == '__main__':
    main()
