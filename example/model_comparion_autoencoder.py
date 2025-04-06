from os import makedirs
from os.path import join
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

from denspp.offline.yaml_handler import YamlConfigHandler
from denspp.offline.dnn.dataset.autoencoder import prepare_training
from denspp.offline.dnn.dnn_handler import ConfigMLPipeline, DefaultSettings_MLPipe
from denspp.offline.dnn.pytorch_config_data import SettingsDataset, DefaultSettingsDataset
from denspp.offline.dnn.pytorch_config_model import ConfigPytorch, DefaultSettingsTrainMSE
from denspp.offline.dnn.plots.plot_dnn import results_training
from denspp.offline.dnn.pytorch_pipeline import do_train_autoencoder


def train_model_normal(used_model_name: str, config_train, config_data, dnn_handler,
                       dataset, ptq_level: list=(12, 8)) -> dict:
    model_stats = dict()
    config_train.model_name = used_model_name

    model_stats['metrics'], model_stats['data_result'], model_stats['path2folder'] = do_train_autoencoder(
        config_ml=dnn_handler, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=config_train.get_model(), calc_custom_metrics=['dsnr_all', 'ptq_loss'],
        print_results=False, ptq_quant_lvl=ptq_level
    )
    used_first_fold = [key for key in model_stats["metrics"].keys()][0]
    results_training(
        path=model_stats["path2folder"], cl_dict=model_stats["data_result"]['cl_dict'],
        feat=model_stats["data_result"]['feat'],
        yin=model_stats["data_result"]['input'], ypred=model_stats["data_result"]['pred'],
        ymean=dataset.get_mean_waveforms,
        yclus=model_stats["data_result"]['valid_clus'],
        snr=model_stats["metrics"][used_first_fold]['dsnr_all'],
        show_plot=dnn_handler.do_block
    )
    return model_stats


def train_model_quantized(used_model_name: str, config_train, config_data, dnn_handler,
                          dataset, ptq_level: list=(12, 8)) -> dict:
    model_stats = dict()
    config_train.model_name = used_model_name
    model_stats['metrics'], model_stats['data_result'], model_stats['path2folder'] = do_train_autoencoder(
        config_ml=dnn_handler, config_data=config_data, config_train=config_train,
        used_dataset=dataset, used_model=config_train.get_model(), calc_custom_metrics=['dsnr_all'],
        print_results=False, ptq_quant_lvl=ptq_level
    )
    used_first_fold = [key for key in model_stats["metrics"].keys()][0]
    results_training(
        path=model_stats["path2folder"], cl_dict=model_stats["data_result"]['cl_dict'], feat=model_stats["data_result"]['feat'],
        yin=model_stats["data_result"]['input'], ypred=model_stats["data_result"]['pred'], ymean=dataset.get_mean_waveforms,
        yclus=model_stats["data_result"]['valid_clus'], snr=model_stats["metrics"][used_first_fold]['dsnr_all'],
        show_plot=dnn_handler.do_block
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
    ax.set_yscale('log')
    ax.set_xlabel('Epoch', fontdict=font)
    ax.set_ylabel('Loss', fontdict=font)
    ax.set_title(label='Performance Comparison (FP vs. QAT (FxP) vs. PTQ (FxP))', fontdict=font)

    if path2save:
        makedirs(path2save, exist_ok=True)
        fig.savefig(join(path2save, 'model_comparison_autoencoder.svg'), format='svg')
    plt.show(block=True)


if __name__ == "__main__":
    # --- Settings
    folder_name = f'../runs'
    train_model_with_batchnorm = True
    used_models = ['CompareDNN_Autoencoder_v1_Torch', 'CompareDNN_Autoencoder_v1_Creator'] if train_model_with_batchnorm else  ['CompareDNN_Autoencoder_woBN_v1_Torch', 'CompareDNN_Autoencoder_woBN_v1_Creator']
    ptq_level = [12, 10]

    # --- Load Configs
    default_hndl = deepcopy(DefaultSettings_MLPipe)
    default_hndl.mode_train_dnn = 3
    default_hndl.do_plot = False
    yaml_handler = YamlConfigHandler(default_hndl, 'config', 'Config_DNN')
    dnn_handler = yaml_handler.get_class(ConfigMLPipeline)

    default_train = DefaultSettingsTrainMSE
    default_train.model_name = used_models[0]
    yaml_nn = YamlConfigHandler(default_train, 'config', f'ConfigAE_Training')
    config_train = yaml_nn.get_class(ConfigPytorch)

    # --- Get Dataset
    default_data = DefaultSettingsDataset
    default_data.data_file_name = 'quiroga'
    default_data.normalization_do = True
    yaml_data = YamlConfigHandler(default_data, 'config', f'ConfigAE_Dataset')
    config_data = yaml_data.get_class(SettingsDataset)
    dataset = prepare_training(settings=config_data, do_classification=False,
                               mode_train_ae=default_hndl.mode_train_dnn, noise_std=default_hndl.autoencoder_noise_std)

    # --- Training
    model_stats_torch = train_model_normal(used_models[0], config_train, config_data, dnn_handler, dataset)
    model_stats_creator = train_model_quantized(used_models[1], config_train, config_data, dnn_handler, dataset)

    # --- Plotting
    loss_torch_train = model_stats_torch['metrics']['fold_000']['loss_train']
    loss_torch_valid = model_stats_torch['metrics']['fold_000']['loss_valid']
    loss_train_ptq = model_stats_torch['metrics']['fold_000']['ptq_loss']
    loss_creator_train = model_stats_creator['metrics']['fold_000']['loss_train']
    loss_creator_valid = model_stats_creator['metrics']['fold_000']['loss_valid']
    plot_model_comparison(loss_torch_train, loss_torch_valid, loss_creator_train, loss_creator_valid, loss_train_ptq, path2save=folder_name)
