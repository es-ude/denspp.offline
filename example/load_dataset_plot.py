from copy import deepcopy
from denspp.offline.data_call.owncloud_handler import OwncloudDownloader
from denspp.offline.dnn.pytorch_config_data import DefaultSettingsDataset
from denspp.offline.dnn.plots.plot_dataset import plot_frames_dataset


if __name__ == "__main__":
    och = OwncloudDownloader(use_dataset=True)
    overview = och.get_overview_data()

    config_test = deepcopy(DefaultSettingsDataset)
    config_test.normalization_do = True
    data = config_test.load_dataset()
    config_test.data_file_name = 'quiroga'

    plot_frames_dataset(data, plot_norm=config_test.normalization_do, plot_show=True, add_subtitle=True)
    print(".done")
