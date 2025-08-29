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
        :param file_name:   String with file name
        :return:            String with file name without data type ending
        """
        used_file_name = [file_name.split(file_end)[0] for file_end in self._ending_chck if file_end in file_name]
        return used_file_name[0] if len(used_file_name) > 0 else file_name

    def write_data_to_csv(self, data: np.ndarray, chapter_line: list, type_load=float) -> None:
        """Writing data from numpy array into csv file
        :param data:            Numpy array with data content
        :param chapter_line:    List with line numbers of chapter data for each column
        :param type_load:       Type of saving and converting numpy content to csv
        :return:                None
        """
        makedirs(self._path2folder, exist_ok=True)
        if len(chapter_line) > 0:
            dimension_data = data.shape[1] if len(data.shape) > 1 else 1
            assert len(chapter_line) == dimension_data, 'The dimension of chapter line must be equal to the number of columns.'
            header = f"{self._delimiter}".join(chapter_line)

            if type_load == str:
                cmds = dict(comments='', header=header, delimiter=self._delimiter, fmt='%s')
            else:
                cmds = dict(comments='', header=header, delimiter=self._delimiter, fmt='%s')
        else:
            if type_load == str:
                cmds = dict(comments='', delimiter=self._delimiter, fmt='%s')
            else:
                cmds = dict(comments='', delimiter=self._delimiter)

        np.savetxt(self.__path2chck, data, **cmds)

    def read_data_from_csv(self, include_chapter_line: bool = False, start_line: int=0, type_load=float) -> np.ndarray:
        """Reading data in numpy format from csv file
        :param include_chapter_line:    Boolean for including the chapter line
        :param start_line:              Number of rows to skip (exclude chapter line)
        :param type_load:               Type of loading and converting csv content to numpy array
        :return:                        Numpy array with data content
        """
        if not exists(self.__path2chck):
            raise FileNotFoundError("CSV does not exists - Please add one!")
        else:
            assert start_line >= 0, "start_line must be larger than 0"
            num_skip_rows = start_line + 1 if include_chapter_line else start_line
            return np.loadtxt(self.__path2chck, delimiter=self._delimiter, skiprows=num_skip_rows, dtype=type_load)

    def read_chapter_from_csv(self, start_line: int=0) -> list:
        """Reading the chapter line in list format from csv file
        :return:    List with chapter lines
        """
        if not exists(self.__path2chck):
            raise FileNotFoundError("CSV does not exists - Please add one!")
        else:
            return np.loadtxt(self.__path2chck, delimiter=self._delimiter, dtype=str, skiprows=start_line, max_rows=1).tolist()
