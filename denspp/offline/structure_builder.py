import logging
from pathlib import Path
from os import makedirs, getcwd
from os.path import join, exists, abspath, dirname
from shutil import copy


logger = logging.getLogger(__name__)


def get_repo_name() -> str:
    """Getting string with repo name"""
    import denspp as ref
    path_to_import = dirname(ref.__file__)
    return Path(path_to_import).parts[-1]


def get_path_to_project_start(new_folder: str='', folder_ref: str='') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:  New folder path (optional)
    :param folder_ref:  Folder reference for finding project path (best case: repo name, optional)
    :return:            String of absolute path to start the project structure
    """
    if get_repo_name() in getcwd() and not folder_ref:
        import denspp as ref
        path_to_import = dirname(ref.__file__)
        path_split = Path(path_to_import).parts[:-1]
        path_to_proj = dirname(join(*[path_seg for path_seg in path_split], ''))
    else:
        path_to_proj = join(getcwd().split(folder_ref)[0], folder_ref) if folder_ref else getcwd()
    path_start = dirname(join(path_to_proj, new_folder))
    logger.debug(f"Project start at: {path_start}")
    return abspath(path_start)


def get_path_to_project() -> str:
    """Function for getting the path to find the project folder structure in application.
    :return:            String of absolute path to start the project structure
    """
    import main_pipeline as ref
    path_to_import = dirname(ref.__file__)
    path_split = Path(path_to_import).parts
    path_to_proj = dirname(join(*[path_seg for path_seg in path_split], ''))
    logger.debug(f"Project start at: {path_to_proj}")
    return abspath(path_to_proj)


def get_path_to_project_templates() -> str:
    """Getting the path to the project templates"""
    import denspp.offline.template as ref
    path_to_temp = dirname(ref.__file__)
    logger.debug(f"Templates available at: {path_to_temp}")
    return abspath(path_to_temp)


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
    folder_structure = ['data', 'dataset', 'runs', 'config', 'src_pipe']
    copy_files = {'main_pipeline.py': '', 'main_data_merge.py': '', 'template_test.py': '', '.gitignore': '',
                  'call_data.py': 'src_pipe', 'pipeline_v0.py': 'src_pipe', 'pipeline_data.py': 'src_pipe'}

    path2start = get_path_to_project_start(new_folder)
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
    path2start = get_path_to_project_start(new_folder)
    for folder_name in folder_structure:
        makedirs(join(path2start, folder_start, folder_name), exist_ok=True)
        if not exists(path2start):
            logger.debug(f"Creating template folder: {folder_name}")

    copy_template_files(copy_files, path2start)
