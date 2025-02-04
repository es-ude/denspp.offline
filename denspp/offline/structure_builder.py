from os import getcwd, makedirs
from os.path import join, exists
from shutil import copy


def get_path_project_start(new_folder: str = '') -> str:
    """Function for getting the path to find the project folder structure.
    :param new_folder:              New folder path (optional)
    :return:                        String of absolute path to start the project structure
    """
    folder_reference = 'denspp.offline'
    folder_start = getcwd().split(folder_reference)[0]
    return join(folder_start, folder_reference, new_folder)


def get_path_to_templates() -> str:
    return join(get_path_project_start(), 'denspp/offline/template')


def init_project_folder(new_folder: str = '') -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_structure = ['data', 'runs', 'temp', 'config', 'src_neuro']
    copy_files = {'main_pipeline.py': '', 'main_data_merge.py': '', 'main_dnn_train.py': '',
                  'call_template.py': 'src_neuro', 'pipeline_v0.py': 'src_neuro'}

    path2start = get_path_project_start(new_folder)
    makedirs(path2start, exist_ok=True)
    if not exists(join(path2start, folder_structure[0])):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    # --- Copy process
    first_element = list(copy_files.items())[0]
    path2test = join(path2start, first_element[1], first_element[0])
    if not exists(path2test):
        path2temp = get_path_to_templates()
        for key, value in copy_files.items():
            copy(join(path2temp, key), join(path2start, value) + '/')


def init_dnn_folder(new_folder: str = '') -> None:
    """Generating a handler dummy for training neural networks
    :param new_folder:      Name of the new folder to create (test case)
    :return:                None
    """
    folder_structure = ['models', 'dataset']
    copy_files = {'main_dnn_train.py': ''}

    # --- Generation process
    path2proj = get_path_project_start(new_folder)
    path2start = join(path2proj, 'src_dnn')
    if not exists(path2start):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    # --- Copy process
    first_element = list(copy_files.items())[0]
    path2test = join(path2proj, first_element[1], first_element[0])
    if not exists(path2test):
        path2temp = get_path_to_templates()
        for key, value in copy_files.items():
            copy(join(path2temp, key), join(path2proj, value) + '/')


if __name__ == '__main__':
    init_project_folder()
    init_dnn_folder()
