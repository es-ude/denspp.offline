from unittest import TestCase, main

from .classifier_dataset import DatasetClassifier
from .dataset_dummy import generate_dummy_dataset


class TestClassifierDataset(TestCase):
    def setUp(self):
        pass

    def test_load_dataset_normal(self):
        dataset = DatasetClassifier(
            dataset=generate_dummy_dataset(100, 10),
        )
        self.assertEqual(dataset.get_cluster_num, 2)
        self.assertEqual(dataset.get_topology_type, "Classifier")
        self.assertEqual(dataset.get_dictionary, ["zero", "one"])
        self.assertEqual(list(dataset[0].keys()), ["in", "out"])


if __name__ == "__main__":
    main()
