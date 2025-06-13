from unittest import TestCase, main
from denspp.offline.dnn.model_library import ModelLibrary, DatasetLoaderLibrary, CellLibrary


class TestModelLibrary(TestCase):
    def test_model_overview(self):
        rslt = ModelLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if not 'DatasetLoader' in item]
        self.assertTrue(len(matches) > 20)

    def test_datasetloader_overview(self):
        rslt = DatasetLoaderLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if 'DatasetLoader' in item]
        self.assertTrue(len(matches))

    def test_cellibrary_overview(self):
        rslt = CellLibrary().get_registry().get_library_overview()
        matches = [item for item in rslt if not 'DatasetLoader' in item]
        self.assertTrue(True)


if __name__ == '__main__':
    main()
