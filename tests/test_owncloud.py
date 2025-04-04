from unittest import TestCase, main
from shutil import rmtree
from os.path import splitext, join, exists
from denspp.offline.data_call.owncloud_handler import ConfigCloud, OwnCloudDownloader
from denspp.offline.structure_builder import get_path_project_start


TestConfigCloud = ConfigCloud(
    remote_link='https://uni-duisburg-essen.sciebo.de/s/JegLJuj1SADBSp0',
    remote_transient='/',
    remote_dataset='/00_Merged',
)


class TestSum(TestCase):
    path2temp = get_path_project_start('temp_test')
    handler = OwnCloudDownloader(
        path2config=path2temp,
        use_config=TestConfigCloud
    )

    def test_access(self):
        overview = self.handler.get_overview_folder(False)
        self.assertGreater(len(overview), 1)

    def test_overview_folder(self):
        overview = self.handler.get_overview_folder(False)
        cnt_dir = sum([1 for folder in overview if folder[-1] == '/'])
        self.assertEqual(len(overview), cnt_dir)

    def test_overview_file(self):
        overview = self.handler.get_overview_data(True)
        cnt_file = sum([1 for file in overview if splitext(file )[-1] == '.npy'])
        self.assertEqual(len(overview), cnt_file)

    def test_download(self):
        overview = self.handler.get_overview_data(True)
        path2file = join(self.path2temp, overview[0].split('/')[-1])

        self.handler.download_file(
            use_dataset=True,
            file_name=overview[0],
            destination_download=path2file
        )
        file_exits = exists(path2file)
        if file_exits:
            rmtree(self.path2temp)
        self.assertTrue(file_exits)


if __name__ == '__main__':
    main()
