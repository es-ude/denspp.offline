import logging
from os import makedirs
from os.path import join, exists
from shutil import copy
from denspp.offline import get_path_to_project, get_path_to_project_templates


logger = logging.getLogger(__name__)


def copy_template_files(copy_files: dict, path2start: str) -> None:
    """Function for copying template files to new folder.
    :param copy_files:          Dictionary of file paths to copy
    :param path2start:          Path to start folder
    :return:                    None
    """
    path2temp = get_path_to_project_templates()
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
    folder_structure = ['data', 'dataset', 'runs', 'config', 'src', 'src_pipe']
    copy_files = {'main_pipeline.py': '', '.gitignore': '', 'README.md': '',
                  'run_tests.py': '', 'call_data.py': 'src_pipe', 'pipeline_plot.py': 'src_pipe',
                  'pipeline_v0.py': 'src_pipe'}
    path2start = get_path_to_project(new_folder)
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
    copy_files = {'main_dnn_train.py': '', 'call_dataset.py': folder_start, 'example_model.py': join(folder_start, 'models')}

    # --- Generation process
    path2start = get_path_to_project(new_folder)
    for folder_name in folder_structure:
        makedirs(join(path2start, folder_start, folder_name), exist_ok=True)
        if not exists(path2start):
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(copy_files, path2start)
