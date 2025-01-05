from os import getcwd, makedirs
from os.path import join, exists
from shutil import copy


def get_path_project_start(start_folder: str = '3_Python', new_folder: str = '') -> str:
    return join(getcwd().split(start_folder)[0], start_folder, new_folder)


def get_path_to_templates() -> str:
    return join(get_path_project_start(), 'package/template')


def init_project_folder(new_folder: str = '') -> None:
    """Generating folder structure in first run
    :param new_folder:      Name of the new folder to create (test case)
    """
    folder_structure = ['data', 'runs', 'test', 'config', 'src_neuro']
    copy_files = {'main_pipeline.py': '', 'call_template.py': 'src_neuro', 'pipeline_v0.py': 'src_neuro'}

    path2start = get_path_project_start(new_folder=new_folder)
    if not exists(join(path2start, folder_structure[0])):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    # --- Copy process
    first_element = list(copy_files.items())[0]
    path2test = join(path2start, first_element[1], first_element[0])
    if not exists(path2test):
        path2temp = join(path2start, 'package/template')
        for key, value in copy_files.items():
            copy(join(path2temp, key), join(path2start, value) + '/')


def init_dnn_folder(new_folder: str = '') -> None:
    """Generating a handler dummy for training neural networks
    :param new_folder:      Name of the new folder to create (test case)
    :param dnn_folder:      Name of the DNN folder to create
    """
    folder_structure = ['models', 'dataset']
    copy_files = {'main_dnn_train.py': ''}

    # --- Generation process
    path2proj = get_path_project_start(new_folder=new_folder)
    path2start = join(path2proj, 'src_dnn')
    if not exists(path2start):
        for folder_name in folder_structure:
            makedirs(join(path2start, folder_name), exist_ok=True)

    # --- Copy process
    first_element = list(copy_files.items())[0]
    path2test = join(path2proj, first_element[1], first_element[0])
    if not exists(path2test):
        path2temp = join(path2proj, 'package/template')
        for key, value in copy_files.items():
            copy(join(path2temp, key), join(path2proj, value) + '/')

        raise TypeError("Folders are generated - Please restart the training routine!")


if __name__ == '__main__':
    init_project_folder()
    init_dnn_folder()
