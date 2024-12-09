from os.path import join
from dataclasses import dataclass
import owncloud
from copy import deepcopy
from package.yaml_handler import yaml_config_handler


@dataclass
class Config_Cloud:
    remote_link: str
    remote_transient: str
    remote_dataset: str


DefaultConfigCloud = Config_Cloud(
    remote_link='https://owncloud.com',
    remote_transient='/',
    remote_dataset='/'
)


class owncloudDownloader:
    __oc_handler: owncloud.Client

    def __init__(self, path2config: str = '', use_dataset=False) -> None:
        """Class for handling sciebo repository for getting datasets remotely"""
        yaml_hndl = yaml_config_handler(deepcopy(DefaultConfigCloud), path2config, 'access_cloud')
        config = yaml_hndl.get_class(Config_Cloud)

        self.__public_sciebo_link = config.remote_link
        self.__path2folder_remote = config.remote_transient if not use_dataset else config.remote_dataset

    def get_overview_data(self, formats: list = ('.npy', '.mat', '.csv')) -> list:
        """Getting an overview of available files for downloading"""
        self.__oc_handler = owncloud.Client.from_public_link(self.__public_sciebo_link)
        dict_list = self.__oc_handler.list(self.__path2folder_remote, 1)
        self.__oc_handler.logout()

        files_available = list()
        for file in dict_list:
            for format in formats:
                if format in file.name:
                    files_available.append(file.name)
        return files_available

    def download_file(self, file_name: str, destination_download: str) -> None:
        """Downloading a file from remote server
        Args:
            file_name:  File name (for downloading remote file)
            destination_download:   Folder name to save the data locally
        Return:
            None
        """
        self.__oc_handler = owncloud.Client.from_public_link(self.__public_sciebo_link)
        print("... downloading file from sciebo")
        self.__oc_handler.get_file(join(self.__path2folder_remote, file_name), destination_download)
        print("... download done")

    def close(self) -> None:
        self.__oc_handler.logout()
