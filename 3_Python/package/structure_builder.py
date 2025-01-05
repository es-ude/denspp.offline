from os import getcwd, makedirs
from os.path import join, exists
from glob import glob
from shutil import copy


def init_project_folder(start_folder='3_Python', new_folder='') -> None:
    """Generating folder structure in first run
    :param start_folder:    Name of the start folder to find in get absolute path
    :param new_folder:      Name of the new folder to create (test case)
    """
    path2start = join(getcwd().split(start_folder)[0], start_folder, new_folder)
    folder_structure = ['data', 'runs', 'test', 'config']
    for folder_name in folder_structure:
        makedirs(join(path2start, folder_name), exist_ok=True)


def init_dnn_folder(start_folder='3_Python', new_folder='', dnn_folder='src_dnn') -> None:
    """Generating a handler dummy for training neural networks
    :param start_folder:    Name of the start folder to find in get absolute path
    :param new_folder:      Name of the new folder to create (test case)
    :param dnn_folder:      Name of the DNN folder to create
    """
    path2start = join(getcwd().split(start_folder)[0], start_folder)
    path2dst = join(path2start, new_folder, dnn_folder)

    folder_structure = ['models', 'dataset']
    for folder_name in folder_structure:
        makedirs(join(path2dst, folder_name), exist_ok=True)

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
