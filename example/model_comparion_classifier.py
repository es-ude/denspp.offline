from os import makedirs
from os.path import join
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from denspp.offline.data_format.yaml import YamlHandler
from denspp.offline.dnn.dataset.autoencoder import prepare_training
from denspp.offline.dnn.dnn_handler import SettingsMLPipeline, DefaultSettingsMLPipeline
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainCE
from denspp.offline.dnn import train_classifier_routine


def train_model_normal(used_model_name: str, config_train, config_data, dnn_handler,
                       dataset, ptq_level: list=(12, 8)) -> dict:
    model_stats = dict()
    config_train.model_name = used_model_name

    model_stats['metrics'], model_stats['data_result'], _ = train_classifier_routine(
        config_ml=dnn_handler, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=config_train.get_model(),
        print_results=False, ptq_quant_lvl=ptq_level
    )
    return model_stats


def train_model_quantized(used_model_name: str, config_train, config_data, dnn_handler,
                          dataset, ptq_level: list=(12, 8)) -> dict:
    model_stats = dict()
    config_train.model_name = used_model_name
    model_stats['metrics'], model_stats['data_result'], _ = train_classifier_routine(
        config_ml=dnn_handler, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=config_train.get_model(), print_results=False, ptq_quant_lvl=ptq_level
    )
    return model_stats


def plot_model_comparison(loss_torch_train, loss_torch_valid, loss_creator_train, loss_creator_valid, loss_train_ptq,
                          fontsize: int=15, path2save: str='') -> None:
    fig, ax = plt.subplots()
    epochs_ite = np.array([idx + 1 for idx in range(len(loss_torch_train))])
    ax.plot(epochs_ite, loss_torch_train, label='FP32, Training', linestyle='solid', marker='.', color='blue')
    ax.plot(epochs_ite, loss_torch_valid, label='FP32, Validation', linestyle='dotted', marker='.', color='blue')
    ax.plot(epochs_ite, loss_creator_train, label='QAT, Training', linestyle='solid', marker='v', color='red')
    ax.plot(epochs_ite, loss_creator_valid, label='QAT, Validation', linestyle='dotted', marker='v', color='red')
    ax.plot(epochs_ite, loss_train_ptq, label='PTQ, Validation', linestyle='dotted', marker='s', color='green')

    font = {'size': fontsize}
    ax.grid()
    ax.legend(fontsize=font['size'])
    ax.margins(0)
    # ax.set_yscale('log')
    ax.set_ylim([0.95, 1.005])
    ax.set_xlabel('Epoch', fontdict=font)
    ax.set_ylabel('Accuracy', fontdict=font)
    ax.set_title(label='Performance Comparison (FP vs. QAT (FxP) vs. PTQ (FxP))', fontdict=font)

    if path2save:
        makedirs(path2save, exist_ok=True)
        fig.savefig(join(path2save, 'model_comparison_classifier.svg'), format='svg')
    plt.show(block=True)


if __name__ == "__main__":
    # --- Settings
    folder_name = f'../runs'
    used_models = ['CompareDNN_Classifier_v1_Torch', 'CompareDNN_Classifier_v1_Creator']
    ptq_level = [12, 11]

    # --- Load Configs
    default_hndl = deepcopy(DefaultSettingsMLPipeline)
    default_hndl.mode_train_dnn = 2
    default_hndl.do_plot = False
    yaml_handler = YamlHandler(default_hndl, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(SettingsMLPipeline)

    default_train = DefaultSettingsTrainCE
    default_train.model_name = used_models[0]
    yaml_nn = YamlHandler(default_train, 'config', f'ConfigCL_Training')
    config_train = yaml_nn.get_class(ConfigPytorch)

    # --- Get Dataset
    default_data = DefaultSettingsDataset
    default_data.data_file_name = 'quiroga'
    default_data.normalization_do = True
    yaml_data = YamlHandler(default_data, 'config', f'ConfigCL_Dataset')
    config_data = yaml_data.get_class(SettingsDataset)
    dataset = prepare_training(
        rawdata=config_data.rawdata,
        do_classification=True
    )

    # --- Training and Plotting
    model_stats_torch = train_model_normal(used_models[0], config_train, config_data, dnn_handler, dataset, ptq_level)
    model_stats_creator = train_model_quantized(used_models[1], config_train, config_data, dnn_handler, dataset, ptq_level)

    loss_torch_train = model_stats_torch['metrics']['fold_000']['acc_train']
    loss_torch_valid = model_stats_torch['metrics']['fold_000']['acc_valid']
    loss_train_ptq = model_stats_torch['metrics']['fold_000']['ptq_loss']
    loss_creator_train = model_stats_creator['metrics']['fold_000']['acc_train']
    loss_creator_valid = model_stats_creator['metrics']['fold_000']['acc_valid']
    plot_model_comparison(loss_torch_train, loss_torch_valid, loss_creator_train, loss_creator_valid, loss_train_ptq, path2save=folder_name)
