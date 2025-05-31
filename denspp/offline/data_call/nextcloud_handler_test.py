from unittest import TestCase, main
from os.path import splitext, join, exists
from denspp.offline.data_call.nextcloud_handler import ConfigCloud, NextCloudDownloader
from denspp.offline.structure_builder import get_path_to_project_start


TestConfigCloud = ConfigCloud(
    remote_link='https://uni-duisburg-essen.sciebo.de/s/JegLJuj1SADBSp0',
    remote_transient='/',
    remote_dataset='/00_Merged',
)


class TestOwnCloud(TestCase):
    path2temp = get_path_to_project_start('temp_test')
    hndl = NextCloudDownloader(
        path2config=path2temp,
        use_config=TestConfigCloud
    )
    """
    def test_nextcloud_access(self):
        self.hndl.get_overview_folder(
            use_dataset=False,
            search_folder=''
        )
        self.assertTrue(False)
    """


if __name__ == '__main__':
    main()
