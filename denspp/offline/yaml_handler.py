import yaml
from logging import getLogger, Logger
from typing import Any
from os import makedirs
from os.path import join, exists
from denspp.offline import get_path_to_project_start


class YamlHandler:
    __logger: Logger
    __path2yaml: str
    __yaml_name: str
    _data: dict
    _template: Any

    def __init__(self, template: Any | dict, path: str= 'config', file_name: str= 'Config_Train'):
        """Creating a class for handling YAML files
        :param template:      Dummy dataclass with entries or dictionary (is only generated if YAML not exist)
        :param path:          String with path to the folder which has the YAML file [Default: '']
        :param file_name:          String with name of the YAML file [Default: 'Config_Train']
        """
        self.__logger = getLogger(__name__)
        self.__path2yaml = join(get_path_to_project_start(), path)
        self.__yaml_name = self.__remove_ending_from_filename(file_name)
        self._template = template

        makedirs(self.__path2yaml, exist_ok=True)
        if not exists(self.path2chck):
            data2yaml = template if isinstance(template, dict) else self._translate_dataclass_to_dict(template)
            self.write_dict_to_yaml(data2yaml)
            self.__logger.info(f"Create new yaml file in folder: {self.__path2yaml}")

    @property
    def path2chck(self) -> str:
        """Getting the path to the desired YAML file"""
        return join(self.__path2yaml, f"{self.__yaml_name}.yaml")

    @staticmethod
    def __remove_ending_from_filename(file_name: str) -> str:
        """Function for removing data type ending
        :param file_name:   String with file name
        :return:            String with file name without data type ending
        """
        yaml_ending_chck = ['.yaml', '.yml']
        yaml_file_name = file_name
        for yaml_end in yaml_ending_chck:
            if yaml_end in yaml_file_name:
                yaml_file_name = yaml_file_name.split(yaml_end)[0]
                break
        return yaml_file_name

    @staticmethod
    def _translate_dataclass_to_dict(class_content: type) -> dict:
        """Translating all class variables with default values into dict
        :param class_content:   Class content
        :return:                Dict with all variables
        """
        return {key: value for key, value in class_content.__dict__.items()
                if not key.startswith('__') and not callable(key)}

    def __check_scheme_validation(self, template: type | dict, real_file: type | dict) -> bool:
        """Function for validating the key entries from template yaml and real yaml file
        :param template:    Dictionary or class from the template for generating yaml file
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

    def write_dict_to_yaml(self, config_data: dict, print_output: bool=False) -> None:
        """Writing list with configuration sets to YAML file
        Args:
            config_data:    Dict. with configuration
            print_output:   Printing the data in YAML format
        Returns:
            None
        """
        makedirs(self.__path2yaml, exist_ok=True)
        with open(self.path2chck, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)

        if print_output:
            print(yaml.dump(config_data, sort_keys=False))

    def get_dict(self) -> dict:
        """Writing list with configuration sets to YAML file
        :return:    Dict. with configuration
        """
        if not exists(self.path2chck):
            raise FileNotFoundError("YAML does not exists - Please create one!")
        else:
            # --- Reading YAML file
            with open(self.path2chck, 'r') as f:
                config_data = yaml.safe_load(f)
            self.__logger.debug(f"... read YAML file: {self.path2chck}")
            return config_data

    def get_class(self, class_constructor: type):
        """Getting all key inputs from yaml dictionary to a class
        :return:    Settings in specified dataclass
        """
        self._data = self.get_dict()
        self.__check_scheme_validation(self._template, self._data)
        return class_constructor(**self._data)
