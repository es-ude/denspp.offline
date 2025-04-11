from os import getcwd, makedirs
from os.path import join, exists, dirname, abspath
from shutil import copy
import denspp.offline


def get_path_project_start(new_folder: str = '') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:              New folder path (optional)
    :return:                        String of absolute path to start the project structure
    """
    folder_reference = 'denspp.offline'
    folder_start = join(getcwd().split(folder_reference)[0], folder_reference) if folder_reference in getcwd() else getcwd()
    return abspath(join(folder_start, new_folder))


def get_path_to_templates() -> str:
    path2start = dirname(denspp.offline.__file__)
    return join(path2start, 'template')


def copy_template_files(copy_files: dict, path2start: str) -> None:
    """Function for copying template files to new folder.
    :param copy_files:          Dictionary of file paths to copy
    :param path2start:          Path to start folder
    :return:                    None
    """
    first_element = list(copy_files.items())[0]
    path2test = join(path2start, first_element[1], first_element[0])
    if not exists(path2test):
        path2temp = get_path_to_templates()
        for key, value in copy_files.items():
            copy(join(path2temp, key), join(path2start, value) + '/')

def init_project_folder(new_folder: str = '') -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_structure = ['data', 'dataset', 'runs', 'temp', 'config', 'src_pipe']
    copy_files = {'main_pipeline.py': '', 'main_data_merge.py': '', 'main_dnn_train.py': '',
                  'call_data.py': 'src_pipe', 'pipeline_v0.py': 'src_pipe'}

    path2start = get_path_project_start(new_folder)
    makedirs(path2start, exist_ok=True)
    if not exists(join(path2start, folder_structure[0])):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    copy_template_files(copy_files, path2start)


def init_dnn_folder(new_folder: str = '') -> None:
    """Generating a handler dummy for training neural networks
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_start = 'src_dnn'
    folder_structure = ['models', 'dataset']
    copy_files = {'main_dnn_train.py': '', 'call_dataset.py': folder_start}

    # --- Generation process
    path2proj = get_path_project_start(new_folder)
    path2start = join(path2proj, folder_start)
    if not exists(path2start):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    copy_template_files(copy_files, path2proj)
