import yaml
import logging
from typing import Any
from os import makedirs
from os.path import join, exists
from denspp.offline.structure_builder import get_path_to_project_start


class YamlConfigHandler:
    __path2yaml_folder: str
    __yaml_name: str
    _data: dict

    @property
    def path2chck(self) -> str:
        """Getting the path to the desired YAML file"""
        return join(self.__path2yaml_folder, f"{self.__yaml_name}.yaml")

    def __init__(self, yaml_template: Any | dict, path2yaml='config', yaml_name='Config_Train'):
        """Creating a class for handling YAML files
        Args:
            yaml_template:      Dummy dataclass with entries or dictionary (is only generated if YAML not exist)
            path2yaml:          String with path to the folder which has the YAML file [Default: '']
            yaml_name:          String with name of the YAML file [Default: 'Config_Train']
        """
        self.__logger = logging.getLogger(__name__)
        self.__path2yaml_folder = join(get_path_to_project_start(), path2yaml)
        self.__yaml_name = self.__remove_ending_from_filename(yaml_name)

        makedirs(self.__path2yaml_folder, exist_ok=True)
        if not exists(self.path2chck):
            data2yaml = yaml_template if isinstance(yaml_template, dict) else self.translate_dataclass_to_dict(yaml_template)
            self.write_dict_to_yaml(data2yaml)
            self.__logger.info("... created new yaml file in folder!")

        self._data = self.read_yaml_to_dict()
        self.__check_scheme_validation(yaml_template, self._data)

    @staticmethod
    def __remove_ending_from_filename(file_name: str) -> str:
        """Function for removing data type ending
        :param file_name: String with file name
        :return:
            String with file name without data type ending
        """
        yaml_ending_chck = ['.yaml', '.yml']
        yaml_file_name = file_name
        for yaml_end in yaml_ending_chck:
            if yaml_end in yaml_file_name:
                yaml_file_name = yaml_file_name.split(yaml_end)[0]
                break
        return yaml_file_name

    def __check_scheme_validation(self, template: type | dict, real_file: type | dict) -> bool:
        """Function for validating the key entries from template yaml and real yaml file
        :param template:    Dictionary or class from the template for generating yaml file
        :param real_file:   Dictionary from real_file
        :return:
            Boolean decision if both key are equal
        """
        template_used = self.translate_dataclass_to_dict(template) if not isinstance(template, dict) else template
        real_used = self.translate_dataclass_to_dict(real_file) if not isinstance(real_file, dict) else real_file

        equal_chck = template_used.keys() == real_used.keys()
        if not equal_chck:
            raise RuntimeError("Config file not valid! - Please check and remove actual config file!")
        else:
            return template_used.keys() == real_used.keys()

    def list_keys(self) -> None:
        """Printing all keys and values of available content in dict"""
        print("\nPrinting the keys and values of existing data")
        print("=======================================================")
        for key in self._data.keys():
            print(f"{key}: {self._data[key]}")
        print("\n")

    def get_value(self, param: str):
        """Getting the content of a specific key input
        Args:
            param:  String with the input
        Returns:
            Value to corresponding key entry
        """
        return self._data[param]

    def get_class(self, class_constructor: type):
        """Getting all key inputs from yaml dictionary to a class"""
        return class_constructor(**self._data)

    def write_dict_to_yaml(self, config_data: dict, print_output: bool=False) -> None:
        """Writing list with configuration sets to YAML file
        Args:
            config_data:    Dict. with configuration
            print_output:   Printing the data in YAML format
        Returns:
            None
        """
        makedirs(self.__path2yaml_folder, exist_ok=True)
        with open(self.path2chck, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)

        if print_output:
            print(yaml.dump(config_data, sort_keys=False))

    def read_yaml_to_dict(self, print_output: bool=False) -> dict:
        """Writing list with configuration sets to YAML file
        Args:
            print_output:   Printing the data in YAML format
        Returns:
            Dict. with configuration
        """
        if not exists(self.path2chck):
            raise FileNotFoundError("YAML does not exists - Please create one!")
        else:
            # --- Reading YAML file
            with open(self.path2chck, 'r') as f:
                config_data = yaml.safe_load(f)
            self.__logger.debug(f"... read YAML file: {self.path2chck}")

            # --- Printing output
            if print_output:
                print(yaml.dump(config_data, sort_keys=False))
            return config_data

    @staticmethod
    def translate_dataclass_to_dict(class_content: type) -> dict:
        """Translating all class variables with default values into dict"""
        return {key: value for key, value in class_content.__dict__.items()
                if not key.startswith('__') and not callable(key)}
