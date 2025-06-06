import numpy as np
from os.path import exists, join
from unittest import TestCase, main
from denspp.offline import get_repo_name, get_path_to_project
from denspp.offline.csv_handler import CsvHandler


# --- Info: Function have to start with test_*
class TestCSV(TestCase):
    chapter_line = ['A', 'B', 'C', 'D']
    data0 = np.array([[1, 2, 3, 1], [4, 5, 6, 7], [7, 8, 9, 0]])
    data1 = np.array([[1e4, 2., 3, 4], [4, 5, 6, 4.58677], [7, 8, 9, 187486.34]])

    path = join(get_path_to_project('temp_test'))
    file = 'test'
    hndl0 = CsvHandler(
        path=path,
        file_name=file
    )
    hndl1 = CsvHandler(
        path=path,
        file_name=f'{file}.csv'
    )

    def test_build_file_exists_wo_chapter(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_wo"
        )
        hndl.write_data_to_csv(
            data=self.data0,
            chapter_line=[]
        )
        chck = exists(join(self.path, 'test_wo.csv'))
        self.assertTrue(chck)

    def test_build_file_exists_with_chapter(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_with"
        )
        hndl.write_data_to_csv(
            data=self.data0,
            chapter_line=self.chapter_line
        )
        chck = exists(join(self.path, 'test_with.csv'))
        self.assertTrue(chck)

    def test_build_file_with_strings(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_with_str"
        )
        hndl.write_data_to_csv(
            data=np.array(["Apfel", "Banane", "Kirsche"]),
            chapter_line=[],
            type_load=str
        )
        hndl.read_data_from_csv(
            include_chapter_line=False,
            start_line=0,
            type_load=str
        )
        self.assertTrue(True)

    def test_read_file_wo_chapter(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_wo"
        )
        hndl.write_data_to_csv(
            data=self.data0,
            chapter_line=[]
        )
        data = hndl.read_data_from_csv(
            include_chapter_line=False
        )
        np.testing.assert_array_almost_equal(data, self.data0, decimal=8)

    def test_read_file_with_chapter(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_with"
        )
        hndl.write_data_to_csv(
            data=self.data0,
            chapter_line=self.chapter_line
        )
        data = hndl.read_data_from_csv(
            include_chapter_line=True
        )
        np.testing.assert_array_almost_equal(data, self.data0, decimal=8)

    def test_read_file_mixed_numbers(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_mixed"
        )
        hndl.write_data_to_csv(
            data=self.data1,
            chapter_line=self.chapter_line
        )
        data = hndl.read_data_from_csv(
            include_chapter_line=True
        )
        np.testing.assert_array_almost_equal(data, self.data1, decimal=8)

    def test_read_chapter(self):
        hndl = CsvHandler(
            path=self.path,
            file_name=f"{self.file}_mixed"
        )
        hndl.write_data_to_csv(
            data=self.data1,
            chapter_line=self.chapter_line
        )
        chap = hndl.read_chapter_from_csv(
            start_line=0
        )
        self.assertEqual(chap, self.chapter_line)

if __name__ == '__main__':
    main()
