from unittest import TestCase, main
from denspp.offline.dnn.model_library import ModelLibrary


class TestSum(TestCase):
    method = ModelLibrary()

    def test_result_value(self):
        result = self.method.get_registry().get_model_library_overview()
        check = len(result) > 3
        self.assertEqual(check, True)


if __name__ == '__main__':
    main()
