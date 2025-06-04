import csv
import numpy as np
from logging import getLogger, Logger
from os import makedirs
from os.path import join, exists, isabs
from denspp.offline import get_path_to_project


class CsvHandler:
    _logger: Logger
    _ending_chck: list = ['.csv']
    _path2folder: str
    _file_name: str
    _delimiter: str


    def __init__(self, path: str, file_name: str, delimiter: str=';'):
        """Creating a class for handling CSV-files
        :param path:        String with path to the folder which has the CSV file
        :param file_name:   String with name of the CSV file
        :param delimiter:   String with delimiter symbol used in the CSV file
        """
        self._logger = getLogger(__name__)
        self._path2folder = join(get_path_to_project(), path) if not isabs(path) else path
        self._file_name = self.__remove_ending_from_filename(file_name)
        assert len(delimiter) == 1, 'Please add a delimiter symbol.'
        self._delimiter = delimiter

    @property
    def __path2chck(self) -> str:
        """Getting the path to the desired CSV file"""
        return join(self._path2folder, f"{self._file_name}{self._ending_chck[0]}")

    def __remove_ending_from_filename(self, file_name: str) -> str:
        """Function for removing data type ending
        :param file_name: String with file name
        :return:
            String with file name without data type ending
        """
        used_file_name = [file_name.split(file_end)[0] for file_end in self._ending_chck if file_end in file_name]
        return used_file_name[0] if len(used_file_name) > 0 else file_name

    def write_data_to_csv(self, data: np.ndarray, chapter_line: list) -> None:
        """Writing data from numpy array into csv file
        :param data:            Numpy array with data content
        :param chapter_line:    List with line numbers of chapter data for each column
        :return:                None
        """
        makedirs(self._path2folder, exist_ok=True)
        if len(chapter_line) > 0:
            dimension_data = data.shape[1] if len(data.shape) > 1 else 1
            assert len(chapter_line) == dimension_data, 'The dimension of chapter line must be equal to the number of columns.'
            header = f"{self._delimiter} ".join(chapter_line)
            np.savetxt(self.__path2chck, data, comments='', header=header, delimiter=self._delimiter)
        else:
            np.savetxt(self.__path2chck, data, delimiter=self._delimiter)


    def read_data_from_csv(self, include_chapter_line: bool=False) -> np.array:
        """Writing list with configuration sets to JSON file
        :param include_chapter_line:    Boolean for including the chapter line
        :return:    Dict. with configuration
        """
        if not exists(self.__path2chck):
            raise FileNotFoundError("CSV does not exists - Please add one!")
        else:
            num_skip_rows = 1 if include_chapter_line else 0
            data = np.loadtxt(self.__path2chck, delimiter=self._delimiter, skiprows=num_skip_rows)
            return data

    @staticmethod
    def _transform_rawdata_from_csv_to_numpy(data: list) -> np.ndarray:
        """Tranforming the csv data to numpy array"""
        # --- Getting meta information
        num_samples = list()
        for idx, data0 in enumerate(data):
            num_samples.append(len(data0))
        num_samples = np.array(num_samples)
        num_channels = len(data)

        # --- Getting data in right format
        data_used = np.zeros((num_channels, num_samples.min()), dtype=float)
        for idx, data_ch in enumerate(data):
            data_ch0 = list()
            for value in data_ch:
                data_ch0.append(float(value))
            data_used[idx, :] = np.array(data_ch0[0:num_samples.min()])

        return data_used
