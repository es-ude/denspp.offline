from os import mkdir
from os.path import exists, join
from shutil import copy
from datetime import datetime
from scipy.io import savemat


class PipelineCMD:
    path2save: str
    _path2pipe: str

    def __init__(self):
        pass

    def generate_folder(self, path2runs: str, addon: str) -> None:
        """Generating the default folder for saving figures and data"""
        str_datum = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder_name = f'{str_datum}_pipeline{addon}'

        if not exists(path2runs):
            mkdir(path2runs)

        path2save = join(path2runs, folder_name)
        if not exists(path2save):
            mkdir(path2save)

        copy(src=self._path2pipe, dst=path2save)
        self.path2save = path2save

    def save_results(self, name: str, data: dict) -> None:
        """Saving the data with a dictionary"""
        path2data = join(self.path2save, name)
        savemat(path2data, data)
        print(f"... data saved in: {path2data}")
