import yaml
from logging import getLogger, Logger
from typing import Any
from os import makedirs
from os.path import join, exists, isabs
from denspp.offline import get_path_to_project_start


class YamlHandler:
    __logger: Logger
    _ending_chck: list = ['.yaml', '.yml']
    _path2folder: str
    _file_name: str
    _template: Any

    def __init__(self, template: Any | dict, path: str= 'config', file_name: str= 'Config_Train'):
        """Creating a class for handling YAML files
        :param template:      Dummy dataclass with entries or dictionary (is only generated if YAML not exist)
        :param path:          String with path to the folder which has the YAML file [Default: '']
        :param file_name:          String with name of the YAML file [Default: 'Config_Train']
        """
        self.__logger = getLogger(__name__)
        self._path2folder = join(get_path_to_project_start(), path) if not isabs(path) else path
        self._file_name = self.__remove_ending_from_filename(file_name)
        self._template = template

        makedirs(self._path2folder, exist_ok=True)
        if not exists(self.__path2chck):
            self.write_to_yaml()

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

    @staticmethod
    def __translate_dataclass_to_dict(class_content: type) -> dict:
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
        :return:            Boolean decision if both key are equal
        """
        keys_tmplt = self.__translate_dataclass_to_dict(template).keys() if not isinstance(template, dict) else template.keys()
        keys_real = self.__translate_dataclass_to_dict(real_file).keys() if not isinstance(real_file, dict) else real_file.keys()

        equal_chck = keys_tmplt == keys_real
        if not equal_chck:
            list_not0 = [key for key in keys_real if key not in keys_tmplt]
            list_not1 = [key for key in keys_tmplt if key not in [keys_real, list_not0]]
            list_not0.extend(list_not1)
            raise RuntimeError(f"Config file not valid (wrong keys: {list_not0})! - Please check and correct/remove actual config file!")
        else:
            return equal_chck

    def write_to_yaml(self) -> None:
        """Writing template configuration to YAML file
        :return:        None
        """
        data2yaml = self._template if isinstance(self._template, dict) else self.__translate_dataclass_to_dict(self._template)
        self.write_dict_to_yaml(data2yaml)
        self.__logger.info(f"Create new yaml file in folder: {self._path2folder}")

    def write_dict_to_yaml(self, config_data: dict, print_output: bool=False) -> None:
        """Writing list with configuration sets to YAML file
        :param config_data:     Dict. with configuration
        :param print_output:    Printing the data in YAML format
        :return:                None
        """
        makedirs(self._path2folder, exist_ok=True)
        with open(self.__path2chck, 'w') as f:
            yaml.dump(config_data, f, sort_keys=False)

        if print_output:
            print(yaml.dump(config_data, sort_keys=False))

    def get_dict(self) -> dict:
        """Getting the dictionary with configuration sets from YAML file
        :return:    Dict. with configuration
        """
        if not exists(self.__path2chck):
            raise FileNotFoundError("YAML does not exists - Please create one!")
        else:
            # --- Reading YAML file
            with open(self.__path2chck, 'r') as f:
                data = yaml.safe_load(f)
            self.__logger.debug(f"... read YAML file: {self.__path2chck}")
            self.__check_scheme_validation(self._template, data)
            return data

    def get_class(self, class_constructor: type):
        """Getting all key inputs from yaml dictionary to a class
        :return:    Settings in specified dataclass
        """
        data = self.get_dict()
        return class_constructor(**data)
