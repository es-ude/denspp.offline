import numpy as np
from unittest import TestCase, main
from denspp.offline.data_call.call_cellbib import SettingsCellSelector, CellSelector


TestSettings = SettingsCellSelector(
    original_id=dict(zip(['One', 'Two', 'Three', 'Four', 'Five', 'Six'], [0, 1, 2, 3, 4, 5])),
    original_to_reduced=dict(zip(['Gerade', 'Ungerade'], [[0, 2, 4], [1, 3, 5]])),
    original_to_group=dict(zip(['FirstTwo', 'LastTwo'], [[0, 1], [4, 5]])),
    original_to_type=dict(zip(['FirstOne', 'LastOne'], [[0], [5]])),
)


# --- Info: Function have to start with test_*
class TestCellSelector(TestCase):
    label_old = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int)
    data_old = np.random.randn(label_old.shape[0], 100)

    def test_cell_merge_get_id_from_name_mode0(self):
        rslt = CellSelector(TestSettings, 0).get_id_from_key('One')
        self.assertEqual(rslt, 0)

    def test_cell_merge_get_name_mode0(self):
        rslt = CellSelector(TestSettings, 0).get_name_from_id(0)
        self.assertEqual(rslt, 'One')

    def test_cell_merge_get_names_mode0(self):
        rslt = CellSelector(TestSettings, 0).get_label_list()
        self.assertEqual(rslt, ['One', 'Two', 'Three', 'Four', 'Five', 'Six'])

    def test_cell_merge_get_id_single_mode0(self):
        rslt = CellSelector(TestSettings, 0).transform_label_to_id_integer(0)
        self.assertEqual(rslt, 0)

    def test_cell_merge_get_id_array_mode0(self):
        label_new = CellSelector(TestSettings, 0).transform_label_to_id_array(self.label_old)
        chck = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=int)
        np.testing.assert_array_equal(label_new, chck)

    def test_cell_merge_get_data_mode0(self):
        label_new, data_new = CellSelector(TestSettings, 0).transform_data_into_new(self.label_old, self.data_old)
        chck_label = self.label_old
        chck_data = self.data_old
        chck = np.array_equal(label_new, chck_label) and np.array_equal(data_new, chck_data)
        self.assertTrue(chck)

    def test_cell_merge_get_id_from_name_mode1(self):
        rslt = CellSelector(TestSettings, 1).get_id_from_key('Ungerade')
        self.assertEqual(rslt, [1, 3, 5])

    def test_cell_merge_get_name_mode1(self):
        rslt = CellSelector(TestSettings, 1).get_name_from_id(0)
        self.assertEqual(rslt, 'Gerade')

    def test_cell_merge_get_names_mode1(self):
        rslt = CellSelector(TestSettings, 1).get_label_list()
        self.assertEqual(rslt, ['Gerade', 'Ungerade'])

    def test_cell_merge_get_id_single_mode1(self):
        rslt = CellSelector(TestSettings, 1).transform_label_to_id_integer(3)
        self.assertEqual(rslt, 1)

    def test_cell_merge_get_id_array_mode1(self):
        label_new = CellSelector(TestSettings, 1).transform_label_to_id_array(self.label_old)
        chck = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        np.testing.assert_array_equal(label_new, chck)

    def test_cell_merge_get_data_mode1(self):
        label_new, data_new = CellSelector(TestSettings, 1).transform_data_into_new(self.label_old, self.data_old)
        chck_label = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
        chck_data = self.data_old
        chck = np.array_equal(label_new, chck_label) and np.array_equal(data_new, chck_data)
        self.assertTrue(chck)

    def test_cell_merge_get_id_from_name_mode2(self):
        rslt = CellSelector(TestSettings, 2).get_id_from_key('FirstTwo')
        self.assertEqual(rslt, [0, 1])

    def test_cell_merge_get_name_mode2(self):
        rslt = CellSelector(TestSettings, 2).get_name_from_id(4)
        self.assertEqual(rslt, 'LastTwo')

    def test_cell_merge_get_names_mode2(self):
        rslt = CellSelector(TestSettings, 2).get_label_list()
        self.assertEqual(rslt, ['FirstTwo', 'LastTwo'])

    def test_cell_merge_get_id_single_mode2(self):
        rslt = CellSelector(TestSettings, 2).transform_label_to_id_integer(3)
        self.assertEqual(rslt, -1)

    def test_cell_merge_get_id_array_mode2(self):
        label_new = CellSelector(TestSettings, 2).transform_label_to_id_array(self.label_old)
        chck = np.array([0, 0, -1, -1, 1, 1, 0, 0, -1, -1, 1, 1, 0, 0, -1, -1, 1, 1], dtype=int)
        np.testing.assert_array_equal(label_new, chck)

    def test_cell_merge_get_data_mode2(self):
        label_new, data_new = CellSelector(TestSettings, 2).transform_data_into_new(self.label_old, self.data_old)
        chck_label = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int)
        chck_data = self.data_old[[0, 1, 4, 5, 6, 7, 10, 11, 12, 13, 16, 17], :]
        chck = np.array_equal(label_new, chck_label) and np.array_equal(data_new, chck_data)
        self.assertTrue(chck)

    def test_cell_merge_get_id_from_name_mode3(self):
        rslt = CellSelector(TestSettings, 3).get_id_from_key('FirstOne')
        self.assertEqual(rslt, [0])

    def test_cell_merge_get_name_mode3(self):
        rslt = CellSelector(TestSettings, 3).get_name_from_id(5)
        self.assertEqual(rslt, 'LastOne')

    def test_cell_merge_get_names_mode3(self):
        rslt = CellSelector(TestSettings, 3).get_label_list()
        self.assertEqual(rslt, ['FirstOne', 'LastOne'])

    def test_cell_merge_get_id_single_mode3(self):
        rslt = CellSelector(TestSettings, 3).transform_label_to_id_integer(3)
        self.assertEqual(rslt, -1)

    def test_cell_merge_get_id_array_mode3(self):
        label_new = CellSelector(TestSettings, 3).transform_label_to_id_array(self.label_old)
        chck = np.array([0, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, 1, 0, -1, -1, -1, -1, 1], dtype=int)
        np.testing.assert_array_equal(label_new, chck)

    def test_cell_merge_get_data_mode3(self):
        label_new, data_new = CellSelector(TestSettings, 3).transform_data_into_new(self.label_old, self.data_old)
        chck_label = np.array([0, 1, 0, 1, 0, 1], dtype=int)
        chck_data = self.data_old[[0, 5, 6, 11, 12, 17], :]
        chck = np.array_equal(label_new, chck_label) and np.array_equal(data_new, chck_data)
        self.assertTrue(chck)


if __name__ == '__main__':
    main()
