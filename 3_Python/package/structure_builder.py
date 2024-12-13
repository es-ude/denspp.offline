from os import getcwd, makedirs
from os.path import join, exists
from glob import glob
from shutil import copy


def create_folder_general_firstrun(start_folder='3_Python') -> None:
    """Generating folder structure in first run"""
    path2start = join(getcwd().split(start_folder)[0], start_folder)
    folder_structure = ['data', 'runs', 'test', 'config']
    for folder_name in folder_structure:
        makedirs(join(path2start, folder_name), exist_ok=True)


def create_folder_dnn_firstrun(project_start_folder='3_Python', dnn_folder='src_dnn') -> None:
    """Generating a handler dummy for training neural networks"""
    path2start = join(getcwd().split(project_start_folder)[0], project_start_folder)
    path2dst = join(path2start, dnn_folder)

    folder_structure = ['models', 'dataset', 'config']
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
