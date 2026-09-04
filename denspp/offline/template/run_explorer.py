from copy import deepcopy
from torch import nn

from denspp.offline.logger import define_logger_runtime
from denspp.offline.dnn import SettingsExplorer, DefaultSettingsExplorer, ExploreClassifier


# Further information for search space definition: https://github.com/es-ude/elastic-ai.explorer/blob/main/elasticai/explorer/hw_nas/search_space/README.md
if __name__ == "__main__":
    shuffle_data = True
    do_train_conv1d = True

    define_logger_runtime()
    sets: SettingsExplorer = deepcopy(DefaultSettingsExplorer)
    exp = ExploreClassifier(settings=sets)

    data_spec = exp.prepare_data(
        do_shuffle=shuffle_data,
        add_dimension=do_train_conv1d
    )
    exp.load_search_space()
    path2run = exp.run_search(
        dataset=data_spec,
        loss_fn=nn.CrossEntropyLoss(),
    )
    exp.run_full_training(
        path2run=path2run,
        shuffle_data=shuffle_data,
        add_dimension=do_train_conv1d,
    )
