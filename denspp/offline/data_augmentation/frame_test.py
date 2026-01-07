import numpy as np
from unittest import TestCase, main
from denspp.offline.metric import calculate_snr_cluster
from .frame import (
    augmentation_mean_waveform,
    augmentation_changing_position,
    augmentation_reducing_samples,
    calculate_frame_mean,
    generate_zero_frames
)


class TestCalculateWaveform(TestCase):
    def test_generate_zero_frames_float(self):
        rslt = generate_zero_frames(
            frame_size=32,
            num_frames=100,
            snr_range=[0, 20],
            fs=20e3,
            return_int=False
        )
        self.assertEqual(rslt[0].shape, (100, 32))
        np.testing.assert_array_almost_equal(rslt[0], np.zeros_like(rslt[0]), decimal=3)
        self.assertEqual(rslt[1].shape, (100,))
        np.testing.assert_array_equal(rslt[1], np.zeros(shape=(100,)))
        self.assertEqual(rslt[2].shape, (32,))
        np.testing.assert_array_almost_equal(rslt[2], np.zeros(shape=(32,)), decimal=3)

    def test_generate_zero_frames_integer(self):
        rslt = generate_zero_frames(
            frame_size=32,
            num_frames=100,
            snr_range=[20, 40],
            fs=20e3,
            return_int=True
        )
        self.assertEqual(rslt[0].shape, (100, 32))
        np.testing.assert_array_equal(rslt[0], np.zeros_like(rslt[0]))
        self.assertEqual(rslt[1].shape, (100, ))
        np.testing.assert_array_equal(rslt[1], np.zeros(shape=(100, )))
        self.assertEqual(rslt[2].shape, (32, ))
        np.testing.assert_array_equal(rslt[2], np.zeros(shape=(32, )))

    def test_calculate_frames_mean(self):
        frames_zero = generate_zero_frames(
            frame_size=32,
            num_frames=10,
            snr_range=[0, 20],
            fs=20e3,
            return_int=True
        )
        dataset = {
            'data': np.concatenate((frames_zero[0], frames_zero[0] + 4.5)),
            'label': np.concatenate((frames_zero[1], frames_zero[1] + 1)),
        }
        rslt = calculate_frame_mean(
            frames_in=dataset['data'],
            frames_cl=dataset['label'],
            return_int=False
        )
        self.assertEqual(rslt.shape, (2, 32))
        self.assertEqual(rslt[0, :].min(), 0)
        self.assertEqual(rslt[1, :].min(), 4.5)


class TestAugmentationWaveform(TestCase):
    def setUp(self):
        self.frames_zero = generate_zero_frames(
            frame_size=32,
            num_frames=10,
            snr_range=[-20, -10],
            fs=20e3,
            return_int=False
        )

    def test_augmentation_mean_waveform(self):
        dataset = {
            'data': np.concatenate((self.frames_zero[0], self.frames_zero[0][0:4, :] + 4.5), axis=0),
            'label': np.concatenate((self.frames_zero[1], self.frames_zero[1][0:4] + 1), axis=0),
            'mean': np.concatenate((self.frames_zero[2].reshape(1, 32), self.frames_zero[2].reshape(1, 32) + 4.5), axis=0)
        }
        snr = calculate_snr_cluster(
            frames_in=dataset['data'],
            frames_cl=dataset['label'],
            frames_mean=dataset['mean']
        )
        rslt = augmentation_mean_waveform(
            frames_mean=dataset['mean'],
            frames_cl=dataset['label'],
            snr_in=snr,
            num_min_frames=100,
            fs=20e3
        )
        self.assertEqual(rslt['frames'].shape, (186, dataset['mean'].shape[1]))
        self.assertEqual(rslt['id'].shape, (186, ))
        np.testing.assert_array_equal(np.unique(rslt['id'], return_counts=True), (np.array([0, 1]), np.array([90, 96])))

    def test_augmentation_change_position(self):
        dataset = {
            'data': np.concatenate((self.frames_zero[0], self.frames_zero[0] + 4.5), axis=0),
            'label': np.concatenate((self.frames_zero[1], self.frames_zero[1] + 1), axis=0),
            'mean': np.concatenate((self.frames_zero[2].reshape(1, 32), self.frames_zero[2].reshape(1, 32) + 4.5),
                                   axis=0)
        }
        rslt = augmentation_changing_position(
            frames_in=dataset['data'],
            frames_cl=dataset['label'],
            num_min_frames=30,
        )
        self.assertEqual(rslt[0].shape, (80, dataset['mean'].shape[1]))
        self.assertEqual(rslt[1].shape, (80,))
        np.testing.assert_array_equal(np.unique(rslt[1], return_counts=True), (np.array([0, 1]), np.array([40, 40])))

    def test_augmentation_reduce_samples(self):
        dataset = {
            'data': np.concatenate((self.frames_zero[0], self.frames_zero[0] + 4.5), axis=0),
            'label': np.concatenate((self.frames_zero[1], self.frames_zero[1] + 1), axis=0),
            'mean': np.concatenate((self.frames_zero[2].reshape(1, 32), self.frames_zero[2].reshape(1, 32) + 4.5), axis=0)
        }
        rslt = augmentation_reducing_samples(
            frames_in=dataset['data'],
            frames_cl=dataset['label'],
            num_frames=4,
            do_shuffle=False
        )
        self.assertEqual(rslt[0].shape, (8, dataset['mean'].shape[1]))
        self.assertEqual(rslt[1].shape, (8,))
        np.testing.assert_array_equal(np.unique(rslt[1], return_counts=True), (np.array([0, 1]), np.array([4, 4])))


if __name__ == '__main__':
    main()
