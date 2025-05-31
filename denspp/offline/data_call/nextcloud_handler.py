import fnmatch
from os.path import join
from dataclasses import dataclass
from logging import getLogger
from nc_py_api import Nextcloud, NextcloudException
from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.structure_builder import get_path_to_project_start


@dataclass
class ConfigCloud:
    """Configuration class for handling remote control using NextCloud
    Attributes:
        remote_link:        String with shared URL to data storage
        remote_transient:   String with path to transient data
        remote_dataset:     String with path to datasets
    """
    remote_link: str
    remote_transient: str
    remote_dataset: str


DefaultConfigCloud = ConfigCloud(
    remote_link='https://owncloud.com',
    remote_transient='/',
    remote_dataset='/'
)


class NextCloudDownloader:
    __handler: Nextcloud
    __settings: ConfigCloud

    def __init__(self, path2config: str = get_path_to_project_start(), use_config: ConfigCloud = DefaultConfigCloud) -> None:
        """Class for handling sciebo repository for getting datasets remotely
        :param path2config: path to config file
        :param use_config:  Class for handling the owncloud handler
        :return:            None
        """
        self.__logger = getLogger(__name__)
        self.__settings = YamlConfigHandler(use_config, path2config, 'access_cloud').get_class(ConfigCloud)

    def __get_remote_content(self, use_dataset: bool, search_folder: str = '', depth: int=1) -> list:
        """Function for getting the remote content in folder
        :param use_dataset:     whether to download datasets (true) or transient signals (false)
        :param search_folder:   folder to search for remote content
        :param depth:           depth of search
        """
        self.__handler = Nextcloud(
            nextcloud_url=self.__settings.remote_link,
            nc_auth_user = "admin",
            nc_auth_pass = "admin"
        )
        try:
            self.__handler.update_server_info()
        except NextcloudException as e:
            print(e)

        path_selected = join(self.__settings.remote_transient if not use_dataset else self.__settings.remote_dataset, search_folder)
        dict_list = self.__handler.files.listdir(depth=depth)
        return dict_list

    def get_overview_folder(self, use_dataset: bool, search_folder: str = '') -> list:
        """Getting an overview of available folders in selected folder
        :param use_dataset: whether to download datasets (true) or transient signals (false) remotely
        :param search_folder:   Search folder path
        :return:                List of folder paths
        """
        remote_content = self.__get_remote_content(
            use_dataset=use_dataset,
            search_folder=search_folder,
            depth=1
        )
        folder_available = [file.path for file in remote_content if file.file_type == 'dir']
        return folder_available

    def get_overview_data(self, use_dataset: bool, search_folder: str = '', format: str = '*.*') -> list:
        """Getting an overview of available files to download
        :param use_dataset: whether to download datasets (true) or transient signals (false) remotely
        :param search_folder:   Search folder path
        :param format:          File format
        :return:                List with available folders/files from remote
        """
        remote_content = self.__get_remote_content(
            use_dataset=use_dataset,
            search_folder=search_folder,
            depth=1
        )
        files_available = [file.path for file in remote_content if file.file_type == 'file']
        return fnmatch.filter(files_available, '*/'+format)

    def download_file(self, use_dataset: bool, file_name: str, destination_download: str) -> None:
        """Downloading a file from remote server
        :param use_dataset:             whether to download datasets (true) or transient signals (false)
        :param  file_name:              File name (for downloading remote file)
        :param  destination_download:   Folder name to save the data locally
        :return:                        None
        """
        self.__handler = Nextcloud.Client.from_public_link(self.__settings.remote_link)
        self.__logger.info("... downloading file from remote")
        path_selected = self.__settings.remote_transient if not use_dataset else self.__settings.remote_dataset
        self.__handler.get_file(join(path_selected, file_name), destination_download)
        self.__logger.info("... download done")

    def close(self) -> None:
        self.__handler.logout()
