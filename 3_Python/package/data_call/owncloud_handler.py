from os.path import join
from dataclasses import dataclass
import fnmatch
import owncloud
from package.yaml_handler import YamlConfigHandler


@dataclass
class ConfigCloud:
    remote_link: str
    remote_transient: str
    remote_dataset: str


DefaultConfigCloud = ConfigCloud(
    remote_link='https://owncloud.com',
    remote_transient='/',
    remote_dataset='/'
)


class OwncloudDownloader:
    __oc_handler: owncloud.Client

    def __init__(self, path2config: str = '', use_dataset=False) -> None:
        """Class for handling sciebo repository for getting datasets remotely"""
        yaml_hndl = YamlConfigHandler(DefaultConfigCloud, path2config, 'access_cloud')
        config = yaml_hndl.get_class(ConfigCloud)

        self.__public_sciebo_link = config.remote_link
        self.__path2folder_remote = config.remote_transient if not use_dataset else config.remote_dataset

    def __get_remote_content(self, search_folder: str = '', depth: int=1) -> list:
        self.__oc_handler = owncloud.Client.from_public_link(self.__public_sciebo_link)
        path_selected = join(self.__path2folder_remote, search_folder) if search_folder else self.__path2folder_remote
        dict_list = self.__oc_handler.list(path_selected, depth)
        self.__oc_handler.logout()
        return dict_list

    def get_overview_folder(self, search_folder: str = '') -> list:
        """Getting an overview of available folders in selected folder"""
        remote_content = self.__get_remote_content(search_folder)
        folder_available = [file.path for file in remote_content if file.file_type == 'dir']
        return folder_available

    def get_overview_data(self, search_folder: str = '',  format: str = '*.*') -> list:
        """Getting an overview of available files to download"""
        remote_content = self.__get_remote_content(search_folder)
        files_available = [file.path for file in remote_content if file.file_type == 'file']
        return fnmatch.filter(files_available, '*/'+format)

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
