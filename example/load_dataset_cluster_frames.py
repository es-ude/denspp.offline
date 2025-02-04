from copy import deepcopy
from denspp.offline.dnn.pytorch_config_data import DefaultSettingsDataset
from denspp.offline.dnn.plots.plot_dataset import plot_frames_dataset


if __name__ == "__main__":
    config_test = deepcopy(DefaultSettingsDataset)
    config_test.normalization_do = True
    config_test.data_file_name = 'quiroga'

    data = config_test.load_dataset()
    plot_frames_dataset(data, plot_norm=config_test.normalization_do, plot_show=True, add_subtitle=True)
    print(".done")
