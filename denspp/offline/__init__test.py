import unittest
from os.path import join
from denspp.offline import (
    check_key_elements,
    check_elem_unique,
    check_keylist_elements_all,
    check_keylist_elements_any,
    check_string_equal_elements_all,
    check_string_equal_elements_any,
    check_value_range, is_close,
    get_path_to_project,
    get_path_to_project_templates
)


class TestHelpFunction(unittest.TestCase):
    def test_is_close_true(self):
        rslt = is_close(value=95, target=100, tolerance=5)
        self.assertTrue(rslt)

    def test_is_close_false(self):
        rslt = is_close(value=94, target=100, tolerance=5)
        self.assertFalse(rslt)

    def test_check_string_elements_all_true(self):
        elements = ['number', 'true']
        rslt = check_string_equal_elements_all(
            text='is_the_number_true',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_string_elements_all_false(self):
        elements = ['number', 'false']
        rslt = check_string_equal_elements_all(
            text='is_the_number_true',
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_string_elements_any_true(self):
        elements = ['number', 'false']
        rslt = check_string_equal_elements_any(
            text='is_the_number_true',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_string_elements_any_false(self):
        elements = ['zero', 'false']
        rslt = check_string_equal_elements_any(
            text='is_the_number_true',
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_key_elements_true(self):
        elements = ['num', 'ber', 'true']
        rslt = check_key_elements(
            key='true',
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_key_elements_false(self):
        elements = ['num', 'ber', 'false']
        rslt = check_key_elements(
            key='true',
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_keylist_elements_all_empty(self):
        elements = ['beat', 'iful', 'true']
        rslt = check_keylist_elements_all(
            keylist=[],
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_keylist_elements_all_true(self):
        elements = ['beat', 'iful', 'true']
        rslt = check_keylist_elements_all(
            keylist=['beat', 'iful', 'true', 'maybe'],
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_keylist_elements_all_false(self):
        elements = ['beat', 'iful', 'today']
        rslt = check_keylist_elements_all(
            keylist=['beatiful', 'tomorrow'],
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_keylist_elements_any_empty(self):
        elements = ['beat', 'iful', 'true']
        rslt = check_keylist_elements_any(
            keylist=[],
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_keylist_elements_any_true(self):
        elements = ['num', 'ber', 'true']
        rslt = check_keylist_elements_any(
            keylist=['number', 'true', 'berry'],
            elements=elements
        )
        self.assertTrue(rslt)

    def test_check_keylist_elements_any_false(self):
        elements = ['num', 'ber', 'true', 'cherry']
        rslt = check_keylist_elements_any(
            keylist=['number', 'false', 'berry'],
            elements=elements
        )
        self.assertFalse(rslt)

    def test_check_elem_unique_string_true(self):
        elements = ['num', 'ber', 'true']
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_string_false(self):
        elements = ['num', 'ber', 'ber']
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_elem_unique_number_true(self):
        elements = [0, 1, 2]
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_number_false(self):
        elements = [0, 0, 2]
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_elem_unique_list_true(self):
        elements = [[0, 1, 2], [3, 4, 5]]
        rslt = check_elem_unique(elements)
        self.assertTrue(rslt)

    def test_check_elem_unique_list_false(self):
        elements = [[0, 1, 2], [3, 1, 5]]
        rslt = check_elem_unique(elements)
        self.assertFalse(rslt)

    def test_check_value_range_true(self):
        rslt = check_value_range(
            value=1.0,
            range=[0.9, 1.1]
        )
        self.assertTrue(rslt)

    def test_check_value_range_false(self):
        rslt = check_value_range(
            value=0.8,
            range=[0.9, 1.1]
        )
        self.assertFalse(rslt)

    def test_get_path_to_project_wo_ref(self):
        ref = ['denspp', 'offline']
        chck = get_path_to_project()
        rslt = [True for key in ref if key in chck]
        self.assertTrue(sum(rslt) == 2)

    def test_get_path_to_project_with_ref(self):
        ref = ['denspp', 'offline', 'test']
        chck = get_path_to_project(new_folder='test')
        rslt = chck == join(get_path_to_project(), "test")
        self.assertTrue(rslt)

    def test_get_path_to_project_templates(self):
        ref = ['denspp', 'offline', 'template']
        chck = get_path_to_project_templates()
        rslt = check_string_equal_elements_all(chck, ref)
        self.assertTrue(rslt)


if __name__ == '__main__':
    unittest.main()
