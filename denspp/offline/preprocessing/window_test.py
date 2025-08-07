import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.preprocessing.window import SettingsWindow, WindowSequencer


class TestWindowSequencer(TestCase):
    sets = SettingsWindow(
        sampling_rate=10e3,
        window_sec=10e-3,
        overlap_sec=0.1e-3
    )

    def test_settings_length(self):
        self.assertEqual(self.sets.window_length, 100)

    def test_settings_overlap(self):
        self.assertEqual(self.sets.overlap_length, 1)

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
        num_samples = num_trials* set0.window_length
        stimuli = 2*(np.random.random(num_samples) - 0.5)
        rslt = WindowSequencer(set0).slide(stimuli)

        assert rslt.shape == (num_trials, set0.window_length)
        for idx, sequence in enumerate(rslt):
            start_point = idx * set0.window_length
            chck = stimuli[start_point:start_point + set0.window_length]
            np.testing.assert_array_equal(sequence, chck)


if __name__ == '__main__':
    main()
