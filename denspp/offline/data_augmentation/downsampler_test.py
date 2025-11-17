import unittest
import numpy as np

from .downsampler import augmentation_downsampling


def build_random_dataset_labeled(n_ch: int = 2, n_label: int = 4, n_samples: int = 1000, n_window=10000) -> tuple[
    np.ndarray, np.ndarray]:
    time0 = np.linspace(start=0, stop=2 * np.pi, num=n_window, endpoint=True, dtype=np.float32)

    if n_ch == 1:
        data = np.zeros((n_samples, n_ch, n_window))
    else:
        data = np.zeros((n_samples, n_window))
    label = np.random.randint(low=0, high=n_label - 1, size=(n_samples,))
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
        assert label.size == 1000
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
        assert label.size == 1000
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 3

    def test_build_dataset_unlabeled_3d(self):
        data, label = build_random_dataset_unlabeled(
            n_ch=2,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        assert label.size == 1000
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 3

    def test_build_dataset_unlabeled_2d(self):
        data, label = build_random_dataset_unlabeled(
            n_ch=2,
            n_label=4,
            n_samples=1000,
            n_window=100
        )
        assert label.size == 1000
        assert label.min() == 0
        assert label.max() == 3
        assert len(data.shape) == 3


class AugmentationDownsampling(unittest.TestCase):
    def test_build_dataset_labeled_3d(self):
        assert True == False



if __name__ == '__main__':
    unittest.main()
