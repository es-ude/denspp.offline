import json
from typing import Any
from logging import getLogger, Logger
from os import makedirs
from os.path import join, exists
from denspp.offline import get_path_to_project


class JsonHandler:
    _logger: Logger
    _data: dict
    _template: Any
    _path2folder: str
    _file_name: str

    def __init__(self, template: Any | dict, path: str='config', file_name: str='Params'):
        """Creating a class for handling JSON files
        :param template:      Dummy dataclass with entries or dictionary (is only generated if JSON not exist)
        :param path:          String with path to the folder which has the JSON file [Default: '']
        :param file_name:          String with name of the YAML file [Default: 'Config_Train']
        """
        self._logger = getLogger(__name__)
        self._path2folder = join(get_path_to_project(), path)
        self._file_name = self.__remove_ending_from_filename(file_name)
        self._template = template

        makedirs(self._path2folder, exist_ok=True)
        if not exists(self.path2chck):
            data2json = template if isinstance(template, dict) else self._translate_dataclass_to_dict(template)
            self.write_dict_to_json(data2json)
            self._logger.info(f"Create new yaml file in folder: {self._path2folder}")

    @property
    def path2chck(self) -> str:
        """Getting the path to the desired JSON file"""
        return join(self._path2folder, f"{self._file_name}.json")

    @staticmethod
    def _translate_dataclass_to_dict(class_content: type) -> dict:
        """Translating all class variables with default values into dict"""
        return {key: value for key, value in class_content.__dict__.items()
                if not key.startswith('__') and not callable(key)}

    @staticmethod
    def __remove_ending_from_filename(file_name: str) -> str:
        """Function for removing data type ending
        :param file_name: String with file name
        :return:
            String with file name without data type ending
        """
        ending_chck = ['.json']
        used_file_name = file_name
        for yaml_end in ending_chck:
            if yaml_end in used_file_name:
                used_file_name = used_file_name.split(yaml_end)[0]
                break
        return used_file_name

    def __check_scheme_validation(self, template: type | dict, real_file: type | dict) -> bool:
        """Function for validating the key entries from template yaml and real json file
        :param template:    Dictionary or class from the template for generating json file
        :param real_file:   Dictionary from real_file
        :return:
            Boolean decision if both key are equal
        """
        template_used = self._translate_dataclass_to_dict(template) if not isinstance(template, dict) else template
        real_used = self._translate_dataclass_to_dict(real_file) if not isinstance(real_file, dict) else real_file

        equal_chck = template_used.keys() == real_used.keys()
        if not equal_chck:
            raise RuntimeError("Config file not valid! - Please check and remove actual config file!")
        else:
            return template_used.keys() == real_used.keys()

    def write_dict_to_json(self, config_data: dict) -> None:
        """Writing list with configuration sets to JSON file
        Args:
            config_data:    Dict. with configuration
        Returns:
            None
        """
        makedirs(self._path2folder, exist_ok=True)
        with open(self.path2chck, 'w') as f:
            json.dump(config_data, f, sort_keys=False)

    def get_dict(self) -> dict:
        """Writing list with configuration sets to JSON file
        :return:    Dict. with configuration
        """
        if not exists(self.path2chck):
            raise FileNotFoundError("YAML does not exists - Please create one!")
        else:
            # --- Reading YAML file
            with open(self.path2chck, 'r') as f:
                config_data = json.load(f)
            self._logger.debug(f"... read JSON file: {self.path2chck}")
            return config_data

    def get_class(self, class_constructor: type):
        """Getting all key inputs from json dictionary to a class"""
        self._data = self.get_dict()
        self.__check_scheme_validation(self._template, self._data)
        return class_constructor(**self._data)
