from unittest import TestCase, main

from denspp.offline.dnn.models.mnist import mnist_mlp_cl_v0
from denspp.offline.dnn.models.waveforms import waveforms_mlp_cl_v0
from .model_library import ModelLibrary, DatasetLoaderLibrary, CellLibrary


class TestModelLibrary(TestCase):
    def test_model_overview(self):
        rslt = ModelLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if not 'DatasetLoader' in item]
        self.assertTrue(len(matches) > 20)
        self.assertTrue(mnist_mlp_cl_v0.__name__ in matches)

    def test_model_build_object(self):
        rslt = ModelLibrary().get_registry().build_object(mnist_mlp_cl_v0.__name__)
        self.assertEqual(rslt, mnist_mlp_cl_v0)

    def test_model_build(self):
        rslt = ModelLibrary().get_registry().build(mnist_mlp_cl_v0.__name__)
        self.assertEqual(type(rslt), mnist_mlp_cl_v0)

    def test_model_signature_without_inputs(self):
        rslt = ModelLibrary().get_registry().get_signature(mnist_mlp_cl_v0.__name__)
        self.assertEqual(rslt, [])

    def test_model_signature_with_inputs(self):
        rslt = ModelLibrary().get_registry().get_signature(waveforms_mlp_cl_v0.__name__)
        self.assertEqual(rslt, ['input_size', 'output_size'])


class TestDatasetlibrary(TestCase):
    def test_datasetloader_overview(self):
        rslt = DatasetLoaderLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if 'DatasetLoader' in item]
        self.assertTrue(len(matches))


class TestCellLibrary(TestCase):
    def test_cellibrary_overview(self):
        rslt = CellLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if not 'DatasetLoader' in item]
        self.assertTrue(True)


if __name__ == '__main__':
    main()
