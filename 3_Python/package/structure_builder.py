import yaml
from os import mkdir, getcwd
from os.path import join, exists
from glob import glob
from shutil import copy


def _create_folder_general_firstrun() -> None:
    """Generating folder structure in first run"""
    folder2search = '3_Python'
    path2start = join(getcwd().split(folder2search)[0], folder2search)

    folder_structure = ['data', 'runs', 'test', 'config']
    # --- Checking if path to local training handler exists
    for foldername in folder_structure:
        if not exists(join(path2start, foldername)):
            mkdir(join(path2start, foldername))


def _create_folder_dnn_firstrun() -> None:
    """Generating a handler dummy for training neural networks"""
    folder2search = '3_Python'
    path2start = join(getcwd().split(folder2search)[0], folder2search)

    # --- Checking if path to local training handler exists
    path2dst = join(path2start, 'src_dnn')
    if not exists(path2dst):
        mkdir(path2dst)

    folder_structure = ['models', 'dataset', 'config']
    for foldername in folder_structure:
        if not exists(join(path2start, foldername)):
            mkdir(join(path2start, foldername))

    # --- Copy process
    if not exists(path2dst):
        path2src = join(path2start, 'package/dnn/template')
        print("\nGenerating a template for ML training")
        for file in glob(join(path2src, "*.py")):
            print(f"... copied: {file}")
            if "main" in file:
                copy(file, f"")
            else:
                copy(file, f"{path2dst}/")
        print("Please restart the training routine!")


def write_data_to_yaml_file(config_data: dict, filename: str,
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

    # --- Do output
    if print_output:
        yaml_output = yaml.dump(config_data, sort_keys=False)
        print(yaml_output)


def read_yaml_data_to_data(filename: str, path2save='',
                           print_output=False) -> dict:
    """Writing list with configuration sets to YAML file
    Args:
        filename:       YAML filename
        path2save:      Optional setting for destination to save
        print_output:   Printing the data in YAML format
    Returns:
        Dict. with configuration
    """
    out_dict = dict()
    path2yaml = join(path2save, f'{filename}.yaml')
    if not exists(path2yaml):
        print("YAML does not exists - Please create one!")

    with open(path2yaml, 'r') as f:
        out_dict = yaml.safe_load(f)
    return out_dict


class yaml_config_handler:
    __path2yaml: str
    __yaml_name: str
    _data: dict

    @property
    def path2chck(self) -> str:
        return join(self.__path2yaml, f"{self.__yaml_name}.yaml")

    def __init__(self, dummy_yaml: dict, path2yaml='', yaml_name='Config_Train'):
        """"""
        self.__path2yaml = path2yaml
        self.__yaml_name = yaml_name

        if not exists(self.path2chck):
            write_data_to_yaml_file(
                dummy_yaml,
                self.__yaml_name, self.__path2yaml
            )
            print("... created new yaml file in folder!")

        self._data = {}
        self._data = read_yaml_data_to_data(
            self.__yaml_name,
            self.__path2yaml
        )

    def list_keys(self) -> None:
        """"""
        print("\nPrinting the keys and values of existing data")
        print("=======================================================")
        for key in self._data.keys():
            print(key)

    def get_value(self, param: str):
        """"""
        return self._data[param]
