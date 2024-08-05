import yaml
from os import mkdir, getcwd
from os.path import join, exists
from glob import glob
from shutil import copy


def write_data_to_yaml_file(config_data: dict, filename: str, path2save='') -> None:
    """Writing list with configuration sets to YAML file
    Args:
        config_data:    Dict. with configuration
        filename:       YAML filename
        path2save:      Optional setting for destination to save
    Returns:
        None
    """
    path2yaml = join(path2save, f'{filename}.yaml')
    with open(path2yaml, 'w') as f:
        yaml.dump(config_data, f, sort_keys=False)


def read_yaml_data_to_data(filename: str, path2save='') -> dict:
    """Writing list with configuration sets to YAML file
    Args:
        filename:       YAML filename
        path2save:      Optional setting for destination to save
    Returns:
        Dict. with configuration
    """
    out_dict = dict()
    path2yaml = join(path2save, f'{filename}.yaml')

    with open(path2yaml, 'r') as f:
        out_dict = yaml.safe_load(f)
    return out_dict


def _create_folder_general_firstrun() -> None:
    """Generating folder structure in first run"""
    folder2search = '3_Python'
    path2start = join(getcwd().split(folder2search)[0], folder2search)
    # --- Checking if path to local training handler exists
    if not exists(join(path2start, 'data')):
        mkdir(join(path2start, 'data'))
    if not exists(join(path2start, 'runs')):
        mkdir(join(path2start, 'runs'))
    if not exists(join(path2start, 'test')):
        mkdir(join(path2start, 'test'))


def _create_folder_dnn_firstrun() -> None:
    """Generating a handler dummy for training neural networks"""
    folder2search = '3_Python'
    path2start = join(getcwd().split(folder2search)[0], folder2search)
    path2dst = join(path2start, 'src_dnn')
    # --- Checking if path to local training handler exists
    if not exists(path2dst):
        mkdir(path2dst)
    if not exists(join(path2dst, 'models')):
        mkdir(join(path2dst, 'models'))
    if not exists(join(path2dst, 'dataset')):
        mkdir(join(path2dst, 'dataset'))

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


if __name__ == "__main__":
    _create_folder_general_firstrun()
    _create_folder_dnn_firstrun()

    data_wr = {
        'Name': 'John Doe',
        'Position': 'DevOps Engineer',
        'Location': 'England',
        'Age': '26',
        'Experience': {'GitHub': 'Software Engineer', 'Google': 'Technical Engineer', 'Linkedin': 'Data Analyst'},
        'Languages': {'Markup': ['HTML'], 'Programming': ['Python', 'JavaScript', 'Golang']}
    }

    yaml_output = yaml.dump(data_wr, sort_keys=False)
    write_data_to_yaml_file(data_wr, 'output')
    data_rd = read_yaml_data_to_data('output')
    print(f"Is content of dict equal?: {data_wr == data_rd}")