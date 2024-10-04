import os
import numpy as np
from scipy.io import savemat
from package.metric import calculate_snr
from src_neuro.sda.sda_plotting import plot_results_single, plot_results_sweep, plot_histogramm


def characterize_sda_output(spk_pos_can: np.ndarray, spk_pos_true: np.ndarray,
                            spk_pos_sda: np.ndarray, fs: float) -> [float, float, float]:
    # Nomenator: {truth}_{sda}
    cnt_true_right = 0
    cnt_false_right = 0
    cnt_false_wrong = 0
    cnt_true_wrong = 0
    dt_range = int(1e-3 * fs)

    # TODO: Fenstermethode anpassen, um richtige Bestimmung zu machen
    # --- Determine metrics (version 1)
    old_pos = 0
    for idx0, spk_pos in enumerate(spk_pos_can):
        if idx0 == 0:
            range = [0, spk_pos]
        else:
            range = [old_pos, spk_pos]
        old_pos = spk_pos

        check_sda = np.where((spk_pos_sda > range[0]) & (spk_pos_sda <= range[1]))[0]
        check_true = np.where((spk_pos_true > range[0]) & (spk_pos_true <= range[1]))[0]
        if check_sda.size == 0:
            if check_true.size == 0:
                cnt_true_wrong += 1
            else:
                cnt_false_wrong += 1
        else:
            if check_true.size == 0:
                cnt_false_right += 1
            else:
                cnt_true_right += 1

    accuracy = float(cnt_true_right / (cnt_true_right + cnt_false_right + cnt_false_wrong))
    precision = float(cnt_true_right / (cnt_true_right + cnt_false_right))
    recall = float(cnt_true_right / (cnt_true_right + cnt_false_wrong))
    cnt = [[cnt_true_right, cnt_false_right], [cnt_false_wrong, cnt_true_wrong]]

    return accuracy, precision, recall, cnt


def do_single_run(pipeline, data,
                  spk_amp: float, spk_period: float, spk_firing_rate: float, spk_snr_in: float,
                  mode_sda: int, mode_thr: int,
                  path2save="", use_smoothing=False) -> None:
    """Function to perform a single run for spike-detection and frame generation"""
    spk_signal, spk_pos_true, spk_pos_may = data.gen_spike_activity(spk_amp, spk_period, spk_firing_rate, spk_snr_in)
    pipeline.define_sda(mode_sda, mode_thr)
    pipeline.run_preprocess(spk_signal, do_smooth=use_smoothing)

    spk_sda = pipeline.x_sda
    spk_thr = pipeline.x_thr
    spk_in_frames = data.cut_frames(spk_signal, spk_pos_true, spk_period)
    spk_in_mean = np.mean(spk_in_frames, axis=0)
    spk_out_frames = data.cut_frames(pipeline.x_adc, spk_pos_true, spk_period)
    spk_out_mean = np.mean(spk_out_frames, axis=0)

    # --- Post-Processing and SNR calculation (single trial)
    spk_snr_in = np.zeros(shape=(spk_in_frames.shape[0],))
    for idx, frame in enumerate(spk_in_frames[:, ]):
        spk_snr_in[idx] = calculate_snr(frame, spk_in_mean)

    spk_snr_out = np.zeros(shape=(spk_out_frames.shape[0],))
    for idx, frame in enumerate(spk_out_frames[:, ]):
        spk_snr_out[idx] = calculate_snr(frame, spk_out_mean)

    print(f"Results: SNR_in = {np.mean(spk_snr_in): .3f} dB +/- {np.std(spk_snr_in): .3f}")
    print(f"Results: SNR_out = {np.mean(spk_snr_out): .3f} dB +/- {np.std(spk_snr_out): .3f}")
    plot_results_single(data.time, spk_signal, spk_sda, spk_thr, spk_out_frames, pipeline.used_methods, path2save)


def do_method_sweep(pipeline, data, spk_amp: float, spk_period: float,
                    spk_firing_rate: np.ndarray, snr_in: np.ndarray,
                    mode_sda: list, mode_thr: list,
                    path2save="", use_smoothing=False, num_repeat=2) -> None:
    """Function for spike-detection incl. frame generation with sweeping SNR and Firing Rate"""
    # --- Loading the input raw data
    num_ite = 0
    num_runs = len(spk_firing_rate) * len(mode_thr) * len(mode_sda)
    for method_sda in mode_sda:
        for method_thr in mode_thr:
            pipeline.define_sda(method_sda, method_thr)
            acc = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))
            tpr = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))
            fpr = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))

            print(f"\nStart Processing with run {num_ite} of {num_runs}:")
            for idx0, spk_fr in enumerate(spk_firing_rate):
                print(f"... start processing at @FR = {spk_fr: .2f} Hz:")
                for idx1, snr in enumerate(snr_in):
                    acc0 = np.zeros(shape=(num_repeat,))
                    tpr0 = np.zeros(shape=(num_repeat,))
                    fpr0 = np.zeros(shape=(num_repeat,))
                    for idx2 in range(num_repeat):
                        # --- Process the data incl. check if output is zero (spontaneous fail)
                        do_repeat_run = True
                        while do_repeat_run:
                            spk_signal, spk_pos_true, spk_pos_may = data.gen_spike_activity(
                                spk_amp, spk_period, spk_fr, snr
                            )
                            pipeline.run_preprocess(spk_signal, do_smooth=use_smoothing)
                            spk_pos_sda = pipeline.x_pos
                            do_repeat_run = True if spk_pos_sda.size == 0 else False
                        # --- Calculation of Metrics
                        acc0[idx2], tpr0[idx2], fpr0[idx2], _ = characterize_sda_output(
                            spk_pos_may, spk_pos_true, spk_pos_sda, pipeline.fs_dig
                        )
                    # --- Backpropagation
                    acc[idx0, idx1] = np.mean(acc0)
                    tpr[idx0, idx1] = np.mean(tpr0)
                    fpr[idx0, idx1] = np.mean(fpr0)
                    # plot_histogramm(acc0, tpr0, fpr0)
                    print(f"\t... @SNR ={snr: .1f} dB --> "
                          f"ACC ={100 * np.mean(acc0): .2f} (+/-{100 * np.std(acc0): .2f}) %, "
                          f"TPR ={100 * np.mean(tpr0): .2f} (+/-{100 * np.std(tpr0): .2f}) %, "
                          f"FPR ={100 * np.mean(fpr0): .2f} (+/-{100 * np.std(fpr0): .2f}) %")
                num_ite += 1

            # --- Saving results and plotting
            savemat(os.path.join(path2save, f'{pipeline.used_methods}_results.mat'),
                    {"method": pipeline.used_methods,
                     "firing_rate": spk_firing_rate, "snr": snr_in,
                     "dt_acc": acc, "tr_rate": tpr, "fr_rate": fpr})
            plot_results_sweep(spk_firing_rate, snr_in, acc, tpr, fpr, pipeline.used_methods, path2save)


def do_calc_roc(pipeline, data, spk_amp: float, spk_period: float,
                spk_firing_rate: np.ndarray, spk_snr_in: np.ndarray,
                mode_sda: list, mode_thr: list,
                path2save="", use_smoothing=False, num_repeat=2) -> None:
    """Determining the Receiver Operating Characteristic (ROC) curve with sweeping the classification threshold
    (More informations in https://www.iguazio.com/glossary/classification-threshold/)"""
    spk_signal, spk_pos_true, spk_pos_may = data.gen_spike_activity(spk_amp, spk_period, spk_firing_rate, spk_snr_in)
    pipeline.define_sda(mode_sda, mode_thr)
    pipeline.run_preprocess(spk_signal, do_smooth=use_smoothing)

    spk_sda = pipeline.x_sda
    spk_thr = pipeline.x_thr
    spk_in_frames = data.cut_frames(spk_signal, spk_pos_true, spk_period)
    spk_in_mean = np.mean(spk_in_frames, axis=0)
    spk_out_frames = data.cut_frames(pipeline.x_adc, spk_pos_true, spk_period)
    spk_out_mean = np.mean(spk_out_frames, axis=0)
    raise NotImplementedError
