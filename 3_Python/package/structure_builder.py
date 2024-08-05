from os import mkdir, getcwd
from os.path import join, exists
from glob import glob
from shutil import copy


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
