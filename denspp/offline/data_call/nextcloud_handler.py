import fnmatch
from logging import getLogger, Logger
from os.path import join
from dataclasses import dataclass
from nc_py_api import Nextcloud, NextcloudException
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline import get_path_to_project


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
    remote_link='http://nextcloud.com',
    remote_transient='/',
    remote_dataset='/'
)


class NextCloudDownloader:
    _hndl: Nextcloud
    _settings: ConfigCloud
    _logger: Logger

    def __init__(self, path2config: str = get_path_to_project(), use_config: ConfigCloud = DefaultConfigCloud) -> None:
        """Class for handling cloud storage access for getting transient data and datasets using Nextcloud services
        :param path2config: Path to config file
        :param use_config:  Class for handling the owncloud handler
        :return:            None
        """
        self._logger = getLogger(__name__)
        self._settings = YamlHandler(
            template=use_config,
            path=path2config,
            file_name='access_cloud'
        ).get_class(ConfigCloud)
        self.__establish_access()

    def __establish_access(self) -> None:
        """Function for establishing the communication to device"""
        self._hndl = Nextcloud(nextcloud_url=self._settings.remote_link)
        try:
            self._hndl.update_server_info()
        except NextcloudException:
            print(NextcloudException)

    def __get_remote_content(self, use_dataset: bool, search_folder: str = '', depth: int=1) -> list:
        """Function for getting the remote content in folder
        :param use_dataset:     whether to download datasets (true) or transient signals (false)
        :param search_folder:   folder to search for remote content
        :param depth:           depth of search
        """
        path_selected = join(self._settings.remote_transient if not use_dataset else self._settings.remote_dataset, search_folder)
        dict_list = self._hndl.files.listdir(path=path_selected, depth=depth)
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
        path_selected = self._settings.remote_transient if not use_dataset else self._settings.remote_dataset
        path2file = join(path_selected, file_name)
        self._logger.info("... downloading file from remote")
        self._hndl.files.download2stream(path2file, destination_download)
        self._logger.info("... download done")

    def close(self) -> None:
        """Function for closing the connection to remote cloud"""
        pass
