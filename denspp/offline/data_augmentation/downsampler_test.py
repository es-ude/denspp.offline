import unittest
import numpy as np
from .downsampler import augmentation_downsampling


def build_random_dataset_labeled(n_ch: int = 2, n_label: int = 4, n_samples: int = 1000, n_window=10000) -> tuple[
    np.ndarray, np.ndarray]:
    time0 = np.linspace(start=0, stop=2 * np.pi, num=n_window, endpoint=True, dtype=np.float32)
    if n_ch > 1:
        data = np.zeros((n_samples, n_ch, n_window))
    else:
        data = np.zeros((n_samples, n_window))

    data[..., :] = np.sin(time0) + 0.01 * np.random.randn(*data.shape)
    label = np.random.randint(low=0, high=n_label, size=(n_samples,))
    return data, label


def build_random_dataset_unlabeled(n_ch: int = 2, n_label: int = 4, n_samples: int = 1000, n_window=10000) -> tuple[
    np.ndarray, np.ndarray]:
    data, label = build_random_dataset_labeled(n_ch, n_label, n_samples, n_window)
    return data, np.zeros_like(label)


class BuildDatasetDownsampling(unittest.TestCase):
    def test_build_dataset_labeled_3d(self):
        data, label = build_random_dataset_labeled(
            n_ch=2,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 3

    def test_build_dataset_labeled_2d(self):
        data, label = build_random_dataset_labeled(
            n_ch=1,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 2

    def test_build_dataset_unlabeled_3d(self):
        data, label = build_random_dataset_unlabeled(
            n_ch=2,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 0
        assert len(data.shape) == 3

    def test_build_dataset_unlabeled_2d(self):
        data, label = build_random_dataset_unlabeled(
            n_ch=2,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        self.assertEqual(label.size, 1000)
        assert label.min() == 0
        assert label.max() == 0
        assert len(data.shape) == 3


class AugmentationDownsamplingLabeled2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = 5
        cls.windowsize = 200
        cls.data_orig, cls.label_orig = build_random_dataset_unlabeled(
            n_ch=1,
            n_label=2,
            n_samples=cls.samples,
            n_window=cls.windowsize,
        )

    def test_augmentation_level0(self):
        num_level = 1
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=True
        )
        np.testing.assert_array_equal(data_new, self.data_orig)
        np.testing.assert_array_equal(label_new, self.label_orig)

    def test_augmentation_level1_drop(self):
        num_level = 2
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=True
        )
        self.assertEqual(data_new.shape[0], self.data_orig.shape[0])
        self.assertEqual(label_new.size, round(self.label_orig.size))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))

    def test_augmentation_level1_add(self):
        num_level = 2
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=False
        )
        self.assertEqual(data_new.shape[0], round(self.data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(self.label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], self.data_orig[0, 0])
        self.assertEqual(label_new[0], self.label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], self.label_orig[idx])
        for idx in range(round(self.windowsize/num_level)):
            self.assertEqual(data_new[self.samples, idx], self.data_orig[0, num_level * idx + 1])

    def test_augmentation_level2_drop(self):
        num_level = 3
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=True
        )
        self.assertEqual(data_new.shape[0], self.data_orig.shape[0])
        self.assertEqual(label_new.size, round(self.label_orig.size))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))

    def test_augmentation_level2_add(self):
        num_level = 3
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=False
        )
        self.assertEqual(data_new.shape[0], round(self.data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(self.label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], self.data_orig[0, 0])
        self.assertEqual(label_new[0], self.label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], self.label_orig[idx])
        for idx in range(round(self.windowsize / num_level)):
            self.assertEqual(data_new[self.samples, idx], self.data_orig[0, num_level * idx + 1])

    def test_augmentation_level9_drop(self):
        num_level = 10
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=True
        )
        self.assertEqual(data_new.shape[0], self.data_orig.shape[0])
        self.assertEqual(label_new.size, round(self.label_orig.size))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))

    def test_augmentation_level9_add(self):
        num_level = 10
        data_new, label_new = augmentation_downsampling(
            data=self.data_orig,
            label=self.label_orig,
            n_downsampling=num_level,
            drop_samples=False
        )
        self.assertEqual(data_new.shape[0], round(self.data_orig.shape[0] * num_level))
        self.assertEqual(label_new.size, round(self.label_orig.size * num_level))
        self.assertEqual(data_new.shape[-1], round(self.data_orig.shape[-1] / num_level))
        self.assertEqual(data_new[0, 0], self.data_orig[0, 0])
        self.assertEqual(label_new[0], self.label_orig[0])
        for idx in range(self.samples):
            self.assertEqual(label_new[self.samples + idx], self.label_orig[idx])
        for idx in range(round(self.windowsize / num_level)):
            self.assertEqual(data_new[self.samples, idx], self.data_orig[0, num_level * idx + 1])


if __name__ == '__main__':
    unittest.main()
