import yaml
from os import getcwd
from os.path import join, exists


def write_dict_to_yaml(config_data: dict, filename: str,
                       path2save='', print_output=False) -> None:
    """Writing list with configuration sets to YAML file
    Args:
        config_data:    Dict. with configuration
        filename:       YAML filename
        path2save:      Optional setting for destination to save
        print_output:   Printing the data in YAML format
    Returns:
        None
    """
    path2yaml = join(path2save, f'{filename}.yaml')
    with open(path2yaml, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)

    if print_output:
        print(yaml.dump(config_data, sort_keys=False))


def read_yaml_to_dict(filename: str, path2save='',
                      print_output=False) -> dict:
    """Writing list with configuration sets to YAML file
    Args:
        filename:       YAML filename
        path2save:      Optional setting for destination to save
        print_output:   Printing the data in YAML format
    Returns:
        Dict. with configuration
    """
    path2yaml = join(path2save, f'{filename}.yaml')
    if not exists(path2yaml):
        print("YAML does not exists - Please create one!")

    with open(path2yaml, 'r') as f:
        config_data = yaml.safe_load(f)

    if print_output:
        print(yaml.dump(config_data, sort_keys=False))
    return config_data


def translate_dataclass_to_dict(class_content: type) -> dict:
    """Translating all class variables with default values into dict"""
    return {key: value for key, value in class_content.__dict__.items()
            if not key.startswith('__') and not callable(key)}


class yaml_config_handler:
    __path2yaml: str
    __yaml_name: str
    _data: dict

    @property
    def path2chck(self) -> str:
        """Getting the path to the desired YAML file"""
        return join(self.__path2yaml, f"{self.__yaml_name}.yaml")

    def __init__(self, dummy_class: type | dict, path2yaml='config', yaml_name='Config_Train', start_folder='3_Python'):
        """Creating a class for handling YAML files
        Args:
            dummy_class:        Dummy dataclass with entries or dictionary (is only generated if YAML not exist)
            path2yaml:          String with path to the YAML file [Default: '']
            yaml_name:          String with name of the YAML file [Default: 'Config_Train']
            start_folder:       Folder to start looking for configuration folder
        """
        self.__path2yaml = join(getcwd().split(start_folder)[0], start_folder, path2yaml)
        self.__yaml_name = yaml_name

        if not exists(self.path2chck):
            data2yaml = dummy_class if isinstance(dummy_class, dict) else translate_dataclass_to_dict(dummy_class)
            write_dict_to_yaml(data2yaml, self.__yaml_name, self.__path2yaml)
            print("... created new yaml file in folder!")

        self._data = {}
        self._data = read_yaml_to_dict(
            self.__yaml_name,
            self.__path2yaml
        )

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
