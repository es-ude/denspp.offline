import logging
from os import getcwd, makedirs
from os.path import join, exists, dirname, abspath
from shutil import copy
import denspp.offline


logger = logging.getLogger(__name__)


def get_path_project_start(new_folder: str = '') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:              New folder path (optional)
    :return:                        String of absolute path to start the project structure
    """
    folder_reference = 'denspp.offline'
    folder_start = join(getcwd().split(folder_reference)[0], folder_reference) if folder_reference in getcwd() else getcwd()
    logger.debug(f"Folder start defined at {folder_start}")
    return abspath(join(folder_start, new_folder))


def get_path_to_templates() -> str:
    path2start = dirname(denspp.offline.__file__)
    logger.debug(f"Folder start defined at {path2start}")
    return join(path2start, 'template')


def copy_template_files(copy_files: dict, path2start: str) -> None:
    """Function for copying template files to new folder.
    :param copy_files:          Dictionary of file paths to copy
    :param path2start:          Path to start folder
    :return:                    None
    """
    path2temp = get_path_to_templates()
    for file_name, folder_name in copy_files.items():
        src = join(path2temp, file_name)
        dst = join(path2start, folder_name)
        makedirs(dst, exist_ok=True)
        if not exists(join(dst, file_name)):
            copy(src=src, dst=dst)
            logger.debug(f"Copy file from: {src} - to: {dst}")

def init_project_folder(new_folder: str = '') -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_structure = ['data', 'dataset', 'runs', 'temp', 'config', 'src_pipe']
    copy_files = {'main_pipeline.py': '', 'main_data_merge.py': '', 'template_test.py': '', '.gitignore': '',
                  'call_data.py': 'src_pipe', 'pipeline_v0.py': 'src_pipe', 'pipeline_data.py': 'src_pipe'}

    path2start = get_path_project_start(new_folder)
    makedirs(path2start, exist_ok=True)

    for folder_name in folder_structure:
        makedirs(join(path2start, folder_name), exist_ok=True)
        if not exists(join(path2start, folder_name)):
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(copy_files, path2start)
    init_dnn_folder(new_folder=new_folder)


def init_dnn_folder(new_folder: str = '') -> None:
    """Generating a handler dummy for training neural networks
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_start = 'src_dnn'
    folder_structure = ['models', 'dataset']
    copy_files = {'main_dnn_train.py': '', 'call_dataset.py': folder_start}

    # --- Generation process
    path2start = get_path_project_start(new_folder)
    for folder_name in folder_structure:
        makedirs(join(path2start, folder_start, folder_name), exist_ok=True)
        if not exists(path2start):
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(copy_files, path2start)
