import numpy as np
from tqdm import tqdm
from src_mem.pipeline_v2 import Pipeline
from src_mem.generate_waveform_dataset import generate_dataset
from src_mem.memristor_plots import plot_pipeline_feat, show_plots


def input_signal() -> [np.ndarray, np.ndarray]:
    # --- Declaration of input
    t_end = 10e-3
    t0 = np.linspace(0, t_end, num=int(t_end * fs_ana), endpoint=True)
    u_off = 0.0
    u_pp = [0.25, 0.3, 0.1]
    f0 = [1e3, 1.8e3, 2.8e3]
    uinp = np.zeros(t0.shape) + u_off
    for idx, peak_val in enumerate(u_pp):
        uinp += peak_val * np.sin(2 * np.pi * t0 * f0[idx])
    uinn = 0.0

    return uinp, uinn


if __name__ == "__main__":
    fs_ana = 20e3
    num_samples = 250
    do_noise = True
    do_transient_plot = False

    # --- Definition of Dataset
    dataset = generate_dataset([0, 5, 7, 8, 9, 10], num_samples, 2, fs_ana,
                               adding_noise=do_noise, pwr_noise_db=-28.2)
    dataset_names = dataset.class_names

    # --- Define Setting
    n_dim = 3
    u_off = [0.75, 1.25]
    t_dly = [10e-3, 25e-3]
    gain = [1.0, 1.0]
    if n_dim >= 3:
        u_off.append(2.0)
        t_dly.append(15e-3)
        gain.append(1.0)

    # --- Run DUT
    dut = Pipeline(fs_ana, 'R')
    dut.define_time_adc_samp([0.46, 0.7])

    data_mem_feat = list()
    data_label = list()
    num_sample = 0
    for data in tqdm(dataset, ncols=100, desc="Progress:"):
        dut.run(data[0], u_off, t_dly)
        data_label.append(data[1])
        data_mem_feat.append(dut.signals.x_feat)

        if do_transient_plot:
            dut.do_plotting(num_sample)
            num_sample += 1

    # --- Getting feature space and plot
    feat_pro = np.zeros((len(data_mem_feat), data_mem_feat[-1].size), dtype=float)
    for idx, data in enumerate(data_mem_feat):
        feat_pro[idx, ] = data

    plot_pipeline_feat(feat_pro, label=data_label, dict=dataset_names,
                       path2save=dut.path2save)
    show_plots()
    print("Done")
