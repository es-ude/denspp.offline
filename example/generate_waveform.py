import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from denspp.offline import get_path_to_project
from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn import SettingsDataset, DefaultSettingsDataset
from denspp.offline.template.call_dataset import DatasetLoader
from denspp.offline.plot_helper import get_plot_color, save_figure


def get_dataset() -> dict:
    """Loading the dataset
    :return:        Dict with rawdata ['data'], label ['label'] and label names ['dict']
    """
    set0 = deepcopy(DefaultSettingsDataset)
    set0.data_file_name = 'Waveforms'

    settings = YamlHandler(
        template=set0,
        path=f'{get_path_to_project()}/config',
        file_name='Config_WaveformDataset'
    ).get_class(SettingsDataset)
    return DatasetLoader(settings=settings).load_dataset()


def plot_dataset(dataset: dict, num_samples_class: int=1, single_plot: bool=False, show_plot: bool=True) -> None:
    class_id = np.unique(dataset['label'])
    if single_plot:
        fig = plt.figure()
        for idx in class_id:
            val_range = np.argwhere(dataset['label'] == idx).flatten()
            sample_idx = np.unique(np.random.randint(low=val_range[0], high=val_range[-1], size=num_samples_class))
            plt.plot(dataset['data'][sample_idx, :].T, label=dataset['dict'][idx], color=get_plot_color(idx))
            plt.plot(np.mean(dataset['data'][val_range, :], axis=0), label=dataset['dict'][idx], color='k')

        plt.legend()
        plt.xlim([0, dataset['data'].shape[1]])
        plt.xlabel("Sample Position")
        plt.ylabel("Signal value")
        plt.grid()
    else:
        fig, axs = plt.subplots(3, int(np.ceil(class_id.size / 3)), sharex=True)
        use_subplot = [False for _ in range(axs.shape[0] * axs.shape[1])]
        for idx in class_id:
            use_subplot[idx] = True
            num_column = int(np.floor(idx / axs.shape[1]))
            val_range = np.argwhere(dataset['label'] == idx).flatten()
            sample_idx = np.unique(np.random.randint(low=val_range[0], high=val_range[-1], size=num_samples_class))
            axs[num_column, idx % axs.shape[1]].plot(dataset['data'][sample_idx, :].T, label=dataset['dict'][idx], color='gray')
            axs[num_column, idx % axs.shape[1]].plot(np.mean(dataset['data'][val_range, :], axis=0), label=dataset['dict'][idx], color='k')
            axs[num_column, idx % axs.shape[1]].set_title(f'{dataset["dict"][idx]}')
            axs[num_column, idx % axs.shape[1]].axis('off')
        for idx in [i for i, val in enumerate(use_subplot) if not val]:
            num_column = int(np.floor(idx / axs.shape[1]))
            axs[num_column, idx % axs.shape[1]].axis('off')

    plt.tight_layout()
    save_figure(fig, path=f'{get_path_to_project()}/runs', name='overview_waveforms')
    if show_plot:
        plt.show()


def plot_single_waveform(dataset: dict, take_class: str, show_plot: bool=True) -> None:
    assert take_class in dataset['dict'], f"Not there. Adapt to {dataset['dict']}"
    take_id = [idx for idx, key in enumerate(dataset['dict']) if key == take_class][-1]
    take_sample_id = np.argwhere(dataset['label'] == take_id).flatten()
    take_sample = np.random.randint(low=take_sample_id[0], high=take_sample_id[-1], size=1)

    fig = plt.figure()
    plt.plot(np.linspace(0, dataset['data'].shape[1], dataset['data'].shape[1]), dataset['data'][take_sample, :].T, color='gray')
    plt.step(np.linspace(0, dataset['data'].shape[1], dataset['data'].shape[1]), dataset['data'][take_sample, :].T, where='mid', color='k', marker='.', markersize=12)
    plt.axis('off')

    plt.tight_layout()
    save_figure(fig, path=f'{get_path_to_project()}/runs', name='single_waveforms')
    if show_plot:
        plt.show()


if __name__ == "__main__":
    data0 = get_dataset()
    plot_dataset(data0, num_samples_class=10, show_plot=False)
    plot_single_waveform(data0, 'SINE_HALF')

