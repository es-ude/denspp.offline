from unittest import TestCase, main
from denspp.offline.dnn.model_library import ModelLibrary


class TestModelLibrary(TestCase):
    method = ModelLibrary()

    def test_model_overview(self):
        result = self.method.get_registry().get_model_library_overview()
        check = len(result) > 3
        self.assertEqual(check, True)


if __name__ == '__main__':
    main()
