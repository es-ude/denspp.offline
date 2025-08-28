from unittest import TestCase, main
from os.path import splitext, join, exists
from denspp.offline.data_call.nextcloud_handler import ConfigCloud, NextCloudDownloader
from denspp.offline import get_path_to_project


TestConfigCloud = ConfigCloud(
    remote_link="http://uni-duisburg-essen.sciebo.de/s/Qf3WpGfBESnZYfx",
    remote_transient='/',
    remote_dataset='/00_Merged',
)


class TestNextCloud(TestCase):
    path2temp = get_path_to_project('temp_test')
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
