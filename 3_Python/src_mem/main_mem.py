import numpy as np
from tqdm import tqdm
from src_mem.pipeline_v3 import Pipeline
from src_mem.generate_waveform_dataset import generate_dataset
from src_mem.memristor_plots import plot_pipeline_feat, show_plots
from package.analog.dev_noise import noise_awgn
from package.digital.cluster import SettingsCluster, Clustering
from package.plot.plot_metric import plot_confusion
from package.plot.plot_neural import plot_signals_neural_cluster
from package.dnn.dataset.autoencoder import RecommendedDataset_Config, prepare_training as load_neural
from package.dnn.dataset.mnist import DatasetMNIST, prepare_training as load_mnist


set_clustering = SettingsCluster(
    no_cluster=6
)


def input_signal(f_samp: float) -> [np.ndarray, np.ndarray]:
    """"""
    # --- Settings
    t_end = 10e-3
    u_off = 0.0
    u_pp = [0.25, 0.3, 0.1]
    f0 = [1e3, 1.8e3, 2.8e3]

    # --- Declaration of input
    t0 = np.linspace(0, t_end, num=int(t_end * f_samp), endpoint=True)
    uinp = np.zeros(t0.shape) + u_off
    for idx, peak_val in enumerate(u_pp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * f0[idx])
    uinn = 0.0

    return uinp, uinn


def get_dataset(mode: int):
    """Loading the datset"""
    match mode:
        case 0:
            # --- Loading waveforms
            num_samples = 500
            fs_ana = 20e3
            dataset = generate_dataset([0, 5, 7, 8, 9, 10], num_samples, 2, fs_ana,
                                       do_normalize_rms=True,
                                       adding_noise=do_noise, pwr_noise_db=-28.2)
            dataset_names = dataset.class_names
        case 1:
            # --- Loading MNIST
            fs_ana = 20e3
            dataset = load_mnist("data", True, True)
            dataset_names = dataset.frame_dict
        case 2:
            # --- Loading neural data (all samples)
            fs_ana = 20e3
            dataset = load_neural(RecommendedDataset_Config, do_classification=True)
            dataset_names = dataset.frame_dict

    return dataset, dataset_names, fs_ana


def get_params(mode: int, n_dim: int) -> [list, list, list, list]:
    """"""
    match mode:
        case 0:
            u_off = [0.25, 1.25]
            t_dly = [10e-3, 25e-3]
            gain = [1.0, 1.0]
            if n_dim >= 3:
                u_off.append(2.0)
                t_dly.append(15e-3)
                gain.append(1.0)
            t0_adc_sec = [0.46, 0.7]
        case 1:
            u_off = [0.75, 1.25]
            t_dly = [5e-3, 10e-3]
            gain = [1.0, 1.0]
            if n_dim >= 3:
                u_off.append(2.0)
                t_dly.append(2e-3)
                gain.append(1.0)
            t0_adc_sec = [0.02, 0.0392]
        case 2:
            u_off = [-4.0, -3.5]
            t_dly = [0.2e-3, 0.5e-3]
            gain = [1.0, 1.0]
            if n_dim >= 3:
                u_off.append(-2.2)
                t_dly.append(0.7e-3)
                gain.append(1.0)
            t0_adc_sec = [1.6e-3]
    return u_off, t_dly, gain, t0_adc_sec


if __name__ == "__main__":
    n_dim = 3
    mode_dataset = 0
    do_noise = True
    do_transient_plot = False
    do_block_plots = True

    # --- Input Definition
    dataset0, clus_names, fs_ana = get_dataset(mode_dataset)
    u_off, t_dly, gain, tq = get_params(mode_dataset, n_dim)

    # --- Run DUT
    cluster_mod = Clustering(set_clustering)
    dut = Pipeline(fs_ana)
    dut.define_time_adc_samp(tq)

    # plot_signals_neural_cluster(dataset0, path2save=dut.path2save)

    data_mem_feat = list()
    data_label = list()
    num_sample = 0
    for data in tqdm(dataset0, ncols=100, desc="Progress:"):
        if mode_dataset == 0:
            data_in = data[0]
            data_cl = data[1]
        elif mode_dataset == 1:
            data_in = data['in'].flatten()
            data_cl = data['out']
        else:
            data_in = data['in'] / data['in'].max()
            #data_in = data['mean'] / data['mean'].max() + noise_awgn(data['mean'].size, fs_ana, -31.2)
            data_cl = data['out']

        dut.run(data_in, u_off, t_dly, do_sum=do_transient_plot)
        data_label.append(data_cl)
        data_mem_feat.append(dut.signals.x_feat)

        if do_transient_plot:
            dut.do_plotting(num_sample)
            num_sample += 1

    # --- Getting feature space and clustering
    feat_pro = np.zeros((len(data_mem_feat), data_mem_feat[-1].size), dtype=float)
    for idx, data in enumerate(data_mem_feat):
        feat_pro[idx, ] = data

    data_label = np.array(data_label, dtype=int)
    pipe_label = cluster_mod.init_kmeans(feat_pro)

    # --- Plotting
    plot_pipeline_feat(feat_pro, label=data_label, dict=clus_names,
                       path2save=dut.path2save)
    show_plots(do_block_plots)

    pipe_label2 = cluster_mod.sort_pred2label_data(pipe_label, data_label, feat_pro)
    plot_confusion(data_label, pipe_label2, show_accuracy=True, cl_dict=clus_names,
                   path2save=dut.path2save)
    show_plots(do_block_plots)
    print("Done")
