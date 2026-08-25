from copy import deepcopy
from torch import nn

from denspp.offline.logger import define_logger_runtime
from denspp.offline.dnn import SettingsExplorer, DefaultSettingsExplorer, ExploreClassifier


if __name__ == "__main__":
    shuffle_data = True

    define_logger_runtime()
    sets: SettingsExplorer = deepcopy(DefaultSettingsExplorer)
    exp = ExploreClassifier(settings=sets)

    data_spec = exp.prepare_data(
        do_shuffle=shuffle_data
    )
    exp.load_search_space()
    path2run = exp.run_search(
        dataset=data_spec,
        loss_fn=nn.CrossEntropyLoss(),
    )
    exp.run_full_training(
        path2run=path2run,
        shuffle_data=shuffle_data
    )
