import numpy as np
from unittest import TestCase, main
from .dataset_dummy import generate_dummy_dataset


class TestDummyDataset(TestCase):
    def test_build_10_2(self):
        config = (10, 2)
        rslt = generate_dummy_dataset(config[0], config[1])
        assert np.unique(rslt.label).size == 2
        assert len(rslt.dict) == 2
        assert rslt.data.shape == config
        assert rslt.data.shape[0] == rslt.label.size

    def test_build_100_200(self):
        config = (100, 200)
        rslt = generate_dummy_dataset(config[0], config[1])
        assert np.unique(rslt.label).size == 2
        assert len(rslt.dict) == 2
        assert rslt.data.shape == config
        assert rslt.data.shape[0] == rslt.label.size


if __name__ == '__main__':
    main()
