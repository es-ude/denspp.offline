import numpy as np
import matplotlib.pyplot as plt

from denspp.offline.plot_helper import get_textsize_paper
from denspp.offline.dnn import PyTorchTrainer, SettingsTraining
from denspp.offline.dnn.dnn_handler import TrainingResults
from denspp.offline.dnn.models.comparison_quant import (
    CompareDNN_ClassifierTorch_v1,
    CompareDNN_ClassifierTorch_woBN_v1,
    CompareDNN_ClassifierCreator_v1,
    CompareDNN_ClassifierCreator_woBN_v1,
    CompareDNN_AutoencoderTorch_v1,
    CompareDNN_AutoencoderCreator_woBN_v1,
    CompareDNN_AutoencoderCreator_v1,
    CompareDNN_AutoencoderTorch_woBN_v1
)


def plot_model_comparison(results_nrm: TrainingResults, results_qnt: TrainingResults, fold_num: int=0) -> None:
    fold_name = list(results_nrm.metrics.keys())
    fold_sel = fold_name[fold_num]
    use_autoencoder_model = results_nrm.settings['train'].mode_train
    used_metrics_nrm = results_nrm.metrics[fold_sel]
    used_metrics_qnt = results_qnt.metrics[fold_sel]
    loss_name = 'loss' if use_autoencoder_model else 'acc'

    fig, ax = plt.subplots()
    epochs_ite = np.array([idx + 1 for idx in range(results_nrm.settings['model'].num_epochs)])
    ax.plot(epochs_ite, used_metrics_nrm[f'{loss_name}_train'], label='FP32, Training', linestyle='solid', marker='.', color='blue')
    ax.plot(epochs_ite, used_metrics_nrm[f'{loss_name}_valid'], label='FP32, Validation', linestyle='dotted', marker='.', color='blue')
    ax.plot(epochs_ite, used_metrics_qnt[f'{loss_name}_train'], label='QAT, Training', linestyle='solid', marker='v', color='red')
    ax.plot(epochs_ite, used_metrics_qnt[f'{loss_name}_valid'], label='QAT, Validation', linestyle='dotted', marker='v', color='red')
    ax.plot(epochs_ite, used_metrics_qnt[f'ptq_{loss_name}'], label='PTQ, Validation', linestyle='dotted', marker='s', color='green')

    font = {'size': get_textsize_paper()}
    ax.grid()
    ax.legend(fontsize=font['size'])
    ax.margins(0)
    ax.set_title(label='Performance Comparison (FP vs. QAT (FxP) vs. PTQ (FxP))', fontdict=font)
    ax.set_xlabel('Epoch', fontdict=font)
    if use_autoencoder_model:
        ax.set_yscale('log')
        addon = 'ae'
    else:
        ax.set_ylabel('Accuracy', fontdict=font)
        ax.set_yscale('linear')
        addon = 'cl'

    path2save = (results_nrm.path / 'qnt').absolute()
    path2save.mkdir(parents=True, exist_ok=True)
    fig.savefig(path2save / f'model_comparison_{addon}.svg', format='svg')
    plt.show(block=True)


if __name__ == "__main__":
    # --- Settings
    dataset = 'Waveforms'
    folder_name = f'../runs'
    do_normalization = True
    train_autoencoder = False
    train_with_batchnorm = True
    num_epochs = 20
    ptq_level = [8, 6]

    used_models_cl = [CompareDNN_ClassifierTorch_v1.__name__, CompareDNN_ClassifierCreator_v1.__name__] if train_with_batchnorm else [CompareDNN_ClassifierTorch_woBN_v1.__name__, CompareDNN_ClassifierCreator_woBN_v1.__name__]
    used_models_ae = [CompareDNN_AutoencoderTorch_v1.__name__, CompareDNN_AutoencoderCreator_v1.__name__] if train_with_batchnorm else [CompareDNN_AutoencoderTorch_woBN_v1.__name__, CompareDNN_AutoencoderCreator_woBN_v1.__name__]

    # --- Define config
    sets_train = SettingsTraining(
        mode_train=0 if not train_autoencoder else 1,
        do_block=False,
        do_ptq=True,
        ptq_total_bitwidth=ptq_level[0],
        ptq_frac_bitwidth=ptq_level[1]
    )

    # --- Train Full Precision Model
    hndl0 = PyTorchTrainer(
        use_case=dataset,
        settings=sets_train,
        default_model=used_models_ae[0] if train_autoencoder else used_models_cl[0],
    )
    hndl0._settings_model.model_name = used_models_ae[0] if train_autoencoder else used_models_cl[0]
    hndl0._settings_model.num_epochs = num_epochs
    hndl0._settings_data.normalization_do = do_normalization
    metrics_nrm = hndl0.do_training()

    # --- Train quantized Model
    hndl0 = PyTorchTrainer(
        use_case=dataset,
        settings=sets_train,
        default_model=used_models_ae[1] if train_autoencoder else used_models_cl[1],
    )
    hndl0._settings_model.model_name = used_models_ae[1] if train_autoencoder else used_models_cl[1]
    hndl0._settings_model.num_epochs = num_epochs
    hndl0._settings_data.normalization_do = do_normalization
    metrics_qnt = hndl0.do_training(metrics_nrm.path / 'qnt')

    # --- Plot results
    plot_model_comparison(
        metrics_nrm,
        metrics_qnt,
        fold_num=0
    )
