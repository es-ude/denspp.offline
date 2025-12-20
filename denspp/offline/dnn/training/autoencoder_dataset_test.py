from unittest import TestCase, main
from .dataset_dummy import generate_dummy_dataset
from .autoencoder_dataset import DatasetAutoencoder, DatasetFromFile


class TestAutoencoderDataset(TestCase):
    def setUp(self):
        pass

    def test_load_dataset_normal(self):
        dataset = DatasetAutoencoder(
            dataset=generate_dummy_dataset(100, 10),
            noise_std=0,
            mode_train=0
        )
        self.assertEqual(dataset.get_cluster_num, 2)
        self.assertEqual(dataset.get_topology_type, "Autoencoder")
        self.assertEqual(dataset.get_dictionary, ["zero", "one"])
        self.assertEqual(dataset.get_mean_waveforms.shape, (2, 10))
        self.assertEqual(list(dataset[0].keys()), ["in", "out", "class", "mean"])

    def test_load_dataset_mean(self):
        dataset = DatasetAutoencoder(
            dataset=generate_dummy_dataset(100, 10),
            noise_std=0,
            mode_train=1
        )
        self.assertEqual(dataset.get_cluster_num, 2)
        self.assertEqual(dataset.get_topology_type, "(mean) Denoising Autoencoder")
        self.assertEqual(dataset.get_dictionary, ["zero", "one"])
        self.assertEqual(dataset.get_mean_waveforms.shape, (2, 10))
        self.assertEqual(list(dataset[0].keys()), ["in", "out", "class", "mean"])

    def test_load_dataset_random(self):
        dataset = DatasetAutoencoder(
            dataset=generate_dummy_dataset(100, 10),
            noise_std=0,
            mode_train=2
        )
        self.assertEqual(dataset.get_cluster_num, 2)
        self.assertEqual(dataset.get_topology_type, "(random noise) Denoising Autoencoder")
        self.assertEqual(dataset.get_dictionary, ["zero", "one"])
        self.assertEqual(dataset.get_mean_waveforms.shape, (2, 10))
        self.assertEqual(list(dataset[0].keys()), ["in", "out", "class", "mean"])

    def test_load_dataset_gaussian(self):
        dataset = DatasetAutoencoder(
            dataset=generate_dummy_dataset(100, 10),
            noise_std=0,
            mode_train=3
        )
        self.assertEqual(dataset.get_cluster_num, 2)
        self.assertEqual(dataset.get_topology_type, "(gaussian noise) Denoising Autoencoder")
        self.assertEqual(dataset.get_dictionary, ["zero", "one"])
        self.assertEqual(dataset.get_mean_waveforms.shape, (2, 10))
        self.assertEqual(list(dataset[0].keys()), ["in", "out", "class", "mean"])


if __name__ == '__main__':
    main()
