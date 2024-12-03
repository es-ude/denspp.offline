from os.path import join
import owncloud


class ScieboDownloadHandler:
    __oc_handler: owncloud.Client

    def __init__(self, link: str = 'https://uni-duisburg-essen.sciebo.de/s/JegLJuj1SADBSp0',
                 path2folder_remote: str = '/00_Merged/') -> None:
        """Class for handling sciebo repository for getting datasets remotely
        Args:
            link:                   String with link to used owncloud repository
            path2folder_remote:     Used folder on remote source
        Return:
            None
        """
        self.__public_sciebo_link = link
        self.__path2folder_remote = path2folder_remote

    def get_overview_data(self, formats: list = ('.npy', '.mat')) -> list:
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
