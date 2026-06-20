from os.path import exists, join, splitext
from unittest import TestCase, main

from denspp.offline import get_path_to_project
from denspp.offline.data_call.owncloud_handler import ConfigCloud, OwnCloudDownloader

TestConfigCloud = ConfigCloud(
    remote_link="https://uni-duisburg-essen.sciebo.de/s/P7XyBAxy3L52ctY",
    remote_transient="/",
    remote_dataset="/00_Merged",
)


class TestOwnCloud(TestCase):
    path2temp = get_path_to_project(new_folder="temp_test")
    handler = OwnCloudDownloader(path2config=path2temp, use_config=TestConfigCloud)

    def test_access(self):
        overview = self.handler.get_overview_folder(False)
        self.assertGreater(len(overview), 0)

    def test_overview_folder(self):
        overview = self.handler.get_overview_folder(False)
        cnt_dir = sum([1 for folder in overview if folder[-1] == "/"])
        self.assertEqual(len(overview), cnt_dir)

    def test_overview_file(self):
        overview = self.handler.get_overview_data(True)
        cnt_file = sum([1 for file in overview if splitext(file)[-1] == ".npy"])
        self.assertEqual(len(overview), cnt_file)

    def test_download(self):
        overview = self.handler.get_overview_data(True)
        path2dest = join(self.path2temp, "data")
        path2file = join(path2dest, overview[0].split("/")[-1])

        self.handler.download_file(
            use_dataset=True, file_name=overview[0], destination_download=path2file
        )
        file_exists = exists(path2file)
        self.assertTrue(file_exists)


if __name__ == "__main__":
    main()
