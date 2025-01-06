import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from os.path import join, split
from torch import load, rand
from package.plot_helper import save_figure, get_plot_color


def extract_best_model_epoch_number(path2search: str, file_index: str = '*.pt') -> int:
    """Extracting the epoch number of best model available
    Args:
        path2search:    Path for look into and find file
        file_index:     Filename index for looking on
    Return:
        Integer with epoch number
    """
    model_path = glob(join(path2search, file_index))[0]
    epoch_num = split(model_path)[-1].split(".pt")[0].split("_epoch")[-1]
    return int(epoch_num)


def extract_model_params(path2search: str, file_index: str = '*.pt') -> int:
    """Extracting the model parameters from a pre-trained model file
    :param path2search:    Path for look into and find file
    :param file_index:     Filename index for looking on
    :return: Integer with model parameters
    """
    model_path = glob(join(path2search, file_index))[0]
    model_used = load(model_path, weights_only=False)
    num_params = int(sum(p.numel() for p in model_used.parameters()))
    return num_params


def extract_model_output_size(path2search: str, file_index: str = '*.pt') -> int:
    """Extracting the model output size from a pre-trained model file
    :param path2search:    Path for look into and find file
    :param file_index:     Filename index for looking on
    :return: Integer with model output size
    """
    model_path = glob(join(path2search, file_index))[0]
    model_used = load(model_path, weights_only=False)
    data_in = rand(model_used.model_shape)
    data_out = model_used(data_in)
    return data_out[1].shape[-1] if '_ae' in file_index else data_out[0].shape[-1]


def extract_feature_size(metric: dict) -> list:
    """Extracting the feature size from a pre-trained model file
    :param metric:  Dictionary with Metrics to extract feature size from
    :return:        Integer with feature size
    """
    feat_size = list()
    for key in metric.keys():
        feat0 = int(key.split('_')[0][-3:])
        if feat0 not in feat_size:
            feat_size.append(feat0)
    return feat_size


def extract_features_from_metric(metric: dict, index_model: str) -> dict:
    """Extracting the feature size from a pre-trained model file
    :param metric:      Dictionary with Metrics to extract feature size from
    :param index_model: Index model to extract feature size from
    :return:            Integer with feature size
    """
    key_feat = list()
    for key in metric.keys():
        if index_model in key:
            key_feat.append(key)

    key_metrics = list()
    for key in metric[key_feat[0]].keys():
        key_metrics.append(key)

    # --- Load model and get params number
    metric_out = dict()
    for key in key_metrics:
        metric_out.update({key: list()})

    for key0 in key_feat:
        for key1 in key_metrics:
            metric_out[key1].append(metric[key0][key1])

    return metric_out


def extract_data_from_files(path2folder: str, folder_split_symbol='\\') -> dict:
    """Loading the metric data from Autoencoder-Classifier Sweep
    Args:
        path2folder:            Path to the folder in which the metric results are available
        folder_split_symbol:    Symbol for splitting the folder
    Return:
        Dictionary with metrics
    """
    list_folder_runs = glob(f"{path2folder}/*")
    if len(list_folder_runs) == 0:
        raise NotADirectoryError("Data not available - Please check the path!")

    data_metrics = dict()
    for folder in list_folder_runs:
        list_data_numpy = glob(f"{folder}/metric_*.npy")

        for file in list_data_numpy:
            type_data = file.split("_")[-1].split('.')[0]
            feat_size = file.split(folder_split_symbol)[-2].split('_size')[-1]
            get_best_epoch = extract_best_model_epoch_number(folder, f'model_{type_data}*.pt')
            get_model_params = extract_model_params(folder, f'model_{type_data}*.pt')
            get_model_output = extract_model_output_size(folder, f'model_{type_data}*.pt')

            data_loaded = np.load(file, allow_pickle=True).flatten()[0]['fold_000']
            cont_data = dict()
            for key, data in data_loaded.items():
                if not key == 'num_clusters':
                    cont_data.update({key: data[get_best_epoch]})
            cont_data.update({"feat_size": int(feat_size), 'model_params': get_model_params, 'output_size': get_model_output})
            data_metrics.update({f"feat{int(feat_size):03d}_{type_data}": cont_data})

    return processing_metric_data(data_metrics)


def processing_metric_data(metric: dict) -> dict:
    """Function for extracting the metrics for the autoencoder and classifier model
    :param metric:  Dictionary with Metrics to extract feature size from
    :return:        Dictionary with params of each model
    """
    feat_ae = extract_features_from_metric(metric, 'ae')
    feat_cl = extract_features_from_metric(metric, 'cl')
    return {'ae': feat_ae, 'cl': feat_cl, 'feat': feat_ae['feat_size']}


def plot_common_loss(metric: dict, path2save: str='', show_plots: bool=False) -> None:
    """Function for plotting the loss function of both models with sweeping the feature space size
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['loss_train', 'loss_valid']
    keys_cl = ['train_acc', 'valid_acc']

    ## --- Subplot #1: Loss / Acc.
    _, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ln0 = ax1.plot(feat_size, metric['ae'][keys_ae[0]], 'k.-', label='Loss_AE (Train)')
    ln1 = ax1.plot(feat_size, metric['ae'][keys_ae[1]], 'k.--', label='Loss_AE (Valid)')
    ax1.set_ylabel("Loss, Autoencoder", fontsize=14)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(feat_size, metric['cl'][keys_cl[0]], 'r.-', label='Acc._CL (Train)')
    ln3 = ax2.plot(feat_size, metric['cl'][keys_cl[1]], 'r.--', label='Acc._CL (Valid)')
    ax2.set_ylabel("Accuracy, Classifier", fontsize=14)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 7))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

    lns = ln0 + ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ## --- End processing
    ax1.grid()
    ax1.set_xlabel('Feature Size', fontsize=14)
    ax1.set_xticks(feat_size_ticks)
    ax1.set_xlim([feat_size[0], feat_size[-1]])
    plt.tight_layout()
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_common_loss', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_common_params(metric: dict, path2save: str= '', show_plots: bool=False) -> None:
    """Function for plotting the parameter numbers of both models with sweeping the feature space size
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['model_params']
    keys_cl = ['model_params']

    _, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True)
    ln4 = ax1.plot(feat_size, metric['ae'][keys_ae[0]], 'k.-', label='Autoencoder (AE)')
    ax2 = ax1.twinx()
    ln5 = ax2.plot(feat_size, metric['cl'][keys_cl[0]], 'r.-', label='Classifier (CL)')
    ax1.set_ylabel('Model parameters', fontsize=14)
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 7))
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 7))

    lns = ln4 + ln5
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc='best')

    ## --- End processing
    ax1.grid()
    ax1.set_xlabel('Feature Size', fontsize=14)
    ax1.set_xticks(feat_size_ticks)
    ax1.set_xlim([feat_size[0], feat_size[-1]])
    plt.tight_layout()

    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_common_params', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_architecture_violin(metric: dict, path2save: str = '', show_plots: bool=False, label_dict=None) -> None:
    """Function for plotting the architecture violin plot
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :param label_dict:  Dictionary with Labels to extract feature size from
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)
    keys_ae = ['dsnr_cl']
    keys_cl = ['precision']

    num_cluster = metric['cl']['output_size'][-1]
    if label_dict is None:
        label_dict = [f'Neuron #{idx}' for idx in range(num_cluster)]

    for key0, key1 in zip(keys_ae, keys_cl):
        _, axs = plt.subplots(nrows=1, ncols=2, sharex=True)
        data_ae = metric['ae'][key0]
        data_cl = metric['cl'][key1]

        # --- Processing AE Data
        transformed_data_ae_cluster = [[] for idx in range(num_cluster)]
        for data_feat in data_ae:
            for idx, data_cluster in enumerate(data_feat):
                transformed_data_ae_cluster[idx].append(data_cluster)

        for idx, data_boxplot in enumerate(transformed_data_ae_cluster):
            axs[0].violinplot(data_boxplot, showmedians=True, positions=np.array(feat_size)+idx/num_cluster, widths=0.5/num_cluster)
        axs[0].legend(label_dict)
        axs[0].set_ylabel(key0, fontsize=14)

        # --- Processing CL Data
        transformed_data_cl_cluster = np.zeros((num_cluster, len(feat_size)))
        for idx, data_cluster in enumerate(data_cl):
            transformed_data_cl_cluster[:, idx] = data_cluster
        for idx, data_plot in enumerate(transformed_data_cl_cluster):
            axs[1].plot(feat_size, data_plot, f'{get_plot_color(idx)}.-', label=label_dict[idx])
        axs[1].set_ylabel(key1, fontsize=14)
        axs[1].legend()

        ## --- End processing
        for ax in axs:
            ax.grid()
            ax.set_xlabel('Feature Size', fontsize=14)
        axs[0].set_xticks(feat_size_ticks)
        axs[0].set_xlim([feat_size[0]-1, feat_size[-1]+1])
        plt.tight_layout()

    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture', ['svg'])
    if show_plots:
        plt.show(block=True)


def plot_architecture_metrics_isolated(metric: dict, path2save: str = '', show_plots: bool=False, label_dict=None) -> None:
    """Function for plotting the metrics in isolated plots for the autoencoder and classifier model
    :param metric:      Dictionary with Metrics to extract feature size from
    :param path2save:   Path for saving the figure
    :param show_plots:  If True, show the plot
    :param label_dict:  Dictionary with Labels to extract feature size from
    :return:            None
    """
    feat_size = metric['feat']
    feat_size_ticks = feat_size if len(feat_size) < 6 else np.linspace(feat_size[0], feat_size[-1], 11,
                                                                       endpoint=True, dtype=np.uint16)

    keys_ae = ['dsnr_cl']
    keys_cl = ['precision']

    num_cluster = metric['cl']['output_size'][-1]
    if label_dict is None:
        label_dict = [f'Neuron #{idx}' for idx in range(num_cluster)]

    num_rows = 2
    num_cols = 3

    # --- Figure #1: Autoencoder
    for key0 in keys_ae:
        _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, sharex=True)
        data_ae = metric['ae'][key0]
        transformed_data_ae_cluster = [[] for idx in range(num_cluster)]
        transformed_data_ae_median = np.zeros((num_cluster, len(feat_size)))
        for idy, data_feat in enumerate(data_ae):
            for idx, data_cluster in enumerate(data_feat):
                transformed_data_ae_cluster[idx].append(data_cluster)
                transformed_data_ae_median[idx, idy] = np.median(data_cluster)
                pass

        for idx, data_boxplot in enumerate(transformed_data_ae_cluster):
            axs[int(idx / num_cols), idx % num_cols].plot(feat_size, transformed_data_ae_median[idx, :], 'k.--', linewidth=1.0)
            axs[int(idx/num_cols), idx % num_cols].violinplot(data_boxplot, showmedians=True, positions=feat_size)
            axs[int(idx/num_cols), idx % num_cols].set_ylabel(f'{key0} ({label_dict[idx]})', fontsize=14)
            axs[int(idx/num_cols), idx % num_cols].grid()

        ## --- End processing
        axs[1, 1].set_xlabel('Feature Size', fontsize=14)
        axs[0, 0].set_xticks(feat_size_ticks)
        axs[0, 0].set_xlim([feat_size[0] - 0.25, feat_size[-1] + 0.25])

    plt.subplots_adjust(wspace=0.3, hspace=0.05)
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture_ae', ['svg'])

    # --- Figure #2: Classifier
    for key1 in keys_cl:
        _, axs = plt.subplots(nrows=1, ncols=len(keys_cl), sharex=True)
        # --- Processing CL Data
        data_cl = metric['cl'][key1]
        transformed_data_cl_cluster = np.zeros((num_cluster, len(feat_size)))
        for idx, data_cluster in enumerate(data_cl):
            transformed_data_cl_cluster[:, idx] = data_cluster
        for idx, data_plot in enumerate(transformed_data_cl_cluster):
            axs.plot(feat_size, data_plot, f'{get_plot_color(idx)}.-', label=label_dict[idx])
        axs.set_ylabel(key1, fontsize=14)
        axs.legend()
        axs.grid()

        ## --- End processing
        axs.set_xlabel('Feature Size', fontsize=14)
        axs.set_xticks(feat_size_ticks)
        axs.set_xlim([feat_size[0]-0.25, feat_size[-1]+0.25])

    plt.subplots_adjust(wspace=0.3, hspace=0.05)
    if path2save:
        save_figure(plt, path2save, 'sweep_dnn_architecture_cl', ['svg'])
    if show_plots:
        plt.show(block=True)


if __name__ == "__main__":
    avai_exp_runs = ['20241202_155954_dnn_ae_v2_sweep', '20241201_230123_cnn_ae_v4_sweep']
    path2run = f'./../../runs/{avai_exp_runs[1]}'

    data = extract_data_from_files(path2run)
    plot_common_loss(data, path2save=path2run)
    plot_common_params(data, path2save=path2run)
    plot_architecture_metrics_isolated(data, show_plots=True, path2save=path2run)
    print("\n.done")
