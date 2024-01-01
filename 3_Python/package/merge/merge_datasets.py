from collections import defaultdict
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt


# region Functions
def crossval(wave1, wave2):
    result = np.correlate(wave1, wave2, 'full') / (np.sqrt(sum(np.square(wave1))) * np.sqrt(sum(np.square(wave2))))
    return result


def mse_loss(yin, yref):
    result = sum(np.square(yin - yref))
    return result


def calculate_snr(wave_in, wave_mean):
    A = (np.max(wave_mean) - np.min(wave_mean)) ** 2
    B = np.sum((wave_in - wave_mean) ** 2)
    outdB = 10 * np.log10(A / B)
    return outdB


def calc_metric(wave_in, wave_ref):
    maxIn = max(wave_in)
    maxInIndex = wave_in.argmax()
    maxRef = max(wave_ref)
    maxRefIndex = wave_ref.argmax()

    result = []
    result.append(maxInIndex - maxRefIndex)
    result.append(mse_loss(wave_in, wave_ref))
    result.append(np.abs(np.trapz(wave_in[:maxInIndex + 1]) - np.trapz(wave_in[maxInIndex:])))
    result.append(maxInIndex)
    result.append(maxIn)
    return np.array(result)


def plot_results(data_packet_X, data_packet_Y, data_packet_mean, path2fig, name):
    val_snr = []
    for idx, value in enumerate(data_packet_X):
        selX = data_packet_X[value]
        selY = data_packet_Y[value]
        selM = data_packet_mean[value]

        val_snr = np.zeros(len(selX))
        for idy in range(len(selX)):
            val_snr[idy] = calculate_snr(selY[idy, :], selM)

    sizeID = len(data_packet_X)
    if sizeID >= 9:
        SubFig = [2, 6]
    else:
        if sizeID <= 4:
            SubFig = [1, 6]
        else:
            SubFig = [2, 6]
    noSubFig = SubFig[0] * SubFig[1]

    for idy in range(0, int(np.ceil(sizeID / noSubFig))):
        plt.figure(figsize=(16, 8))

        selID = np.arange(0, noSubFig) + noSubFig * idy
        if selID[-1] > sizeID:
            selID = np.arange(selID[0], sizeID)

        IteNo = 1
        for idx in range(0, len(selID)):
            NoCluster = selID[idx]

            # Decision if more than 100 frames
            Yin = data_packet_Y[NoCluster]
            if Yin.shape[0] > 2000:
                selFrames = np.random.choice(Yin.shape[0], 2000, replace=False)
            else:
                selFrames = np.arange(0, Yin.shape[0])

            val_snr0 = np.zeros(Yin.shape[0])
            for idz in range(Yin.shape[0]):
                val_snr0[idz] = calculate_snr(Yin[idz, :], np.mean(Yin, axis=0))

            # Plot
            formatSpec = '{:.2f}'
            plt.subplot(SubFig[0], SubFig[1], IteNo)
            IteNo += 1

            plt.plot(Yin[selFrames].T, 'b')
            plt.grid(True)
            plt.plot(np.mean(Yin, axis=0), 'r', linewidth=2)

            plt.title(
                "Cluster ID: " + str(NoCluster) + "\n" + " SNR = " + formatSpec.format(np.mean(val_snr0)) + " (\pm" +
                formatSpec.format(np.std(val_snr0)) + ")\ndB - No = " + str(Yin.shape[0]))

            plt.xlabel("Frames")
            plt.ylabel("Amplitude")
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

        plt.tight_layout()
        plt.savefig(path2fig + name + str(idy).zfill(2) + '.jpg')
        plt.close()


# endregion


def merge_datasets(path_2_file=""):
    # region Pre-Processing: Input structuring
    setOptions = dict()
    setOptions['do_2nd_run'] = False
    setOptions['do_resort'] = True
    addon = '_Sorted'
    root = tk.Tk()
    filetype = (('MAT-Files', '*.mat'),)
    root.withdraw()
    if not path_2_file:
        path_2_file = filedialog.askopenfilename(title='Select file to merge', filetypes=filetype)
    setOptions['path2file'] = path_2_file
    setOptions['path2save'] = path_2_file[:len(path_2_file) - 4] + addon + '.mat'
    setOptions['path2fig'] = path_2_file[:len(path_2_file) - 4]

    mat_file = loadmat(path_2_file)

    if "Martinez" in path_2_file:
        # Settings Martinez
        criterion_CheckDismiss = [3, 0.7]
        criterion_Run0 = 0.98
        criterion_Resort = 0.98
    elif "Quiroga" in path_2_file:
        # Settings für Quiroga
        criterion_CheckDismiss = [2, 0.96]
        criterion_Run0 = 0.98
        criterion_Resort = 0.95
    else:
        criterion_CheckDismiss = [3, 0.7]
        criterion_Run0 = 0.98
        criterion_Resort = 0.98

    frames_cluster = mat_file['frames_cluster']
    frames_in = mat_file['frames_in']
    frames_in_number = frames_in.shape[0]
    print("Start of merging the dataset to one")

    if frames_cluster.shape[0] == 1:
        frames_cluster = np.transpose(frames_cluster)

    frames_cluster = frames_cluster.tolist()
    frames_cluster = [item for sublist in frames_cluster for item in sublist]

    input_cluster = list(set(frames_cluster))  # keeps order
    data_raw_pos = defaultdict()  # data_raw{2}
    data_raw_frames = defaultdict()  # data_raw{3}
    data_raw_means = defaultdict()  # data_raw{4}
    data_raw_metric = defaultdict()  # data_raw{5}
    data_raw_number = 0

    for value in input_cluster:
        pos_in = np.array([i for i, val in enumerate(frames_cluster) if val == value])
        data_raw_pos[value] = pos_in
        data_raw_frames[value] = frames_in[pos_in, :]
        data_raw_means[value] = np.mean(frames_in[pos_in, :], axis=0, dtype=np.float64)
        YCheck = data_raw_frames[value]
        WaveRef = crossval(np.mean(YCheck, axis=0, dtype=np.float64), np.mean(YCheck, axis=0, dtype=np.float64))
        metric_Check = [calc_metric(crossval(YCheck[idy, :], np.mean(YCheck, axis=0, dtype=np.float64)), WaveRef) for
                        idy in range(len(pos_in))]
        data_raw_metric[value] = metric_Check
        data_raw_number += len(pos_in)

    print('... data loaded and pre-selected')
    del frames_in, frames_cluster, value, pos_in, YCheck, WaveRef, metric_Check, mat_file
    # endregion

    # region Pre-Processing: Consistency Check
    data_process_XCheck = defaultdict()  # data_1process[2]
    data_process_YCheck = defaultdict()  # data_1process[3]
    data_process_mean = defaultdict()  # data_1process[4]
    data_process_metric = defaultdict()  # data_1process[5]
    data_dismiss_XCheck = defaultdict()
    data_dismiss_YCheck = defaultdict()
    data_dismiss_mean = defaultdict()
    data_dismiss_metric = defaultdict()

    for idx in input_cluster:
        do_run = True
        YCheckIn = data_raw_frames[idx]
        XCheckIn = data_raw_pos[idx]
        XCheck = np.arange(0, len(XCheckIn))
        XCheck_False = []

        IteNo = 0
        while do_run and len(XCheck) > 0:
            YCheck = YCheckIn[XCheck, :]
            metric_Check = list()
            mean_wfg = np.mean(YCheck, axis=0, dtype=np.float64)
            WaveRef = crossval(mean_wfg, mean_wfg)
            for idy, Y in enumerate(YCheck):
                WaveIn = crossval(Y, mean_wfg)
                calc_temp = np.append(calc_metric(WaveIn, WaveRef), XCheck[idy])
                metric_Check.append(calc_temp)

            metric_Check = np.array(metric_Check)
            criteria = (np.abs(metric_Check[:, 0]) > criterion_CheckDismiss[0]) | (
                    metric_Check[:, 4] < criterion_CheckDismiss[1])
            check = np.where(criteria)[0]
            XCheck_False.extend(XCheck[check])
            if len(check) > 0 and IteNo <= 100:
                do_run = True
                IteNo = IteNo + 1
            else:
                do_run = False
                IteNo = IteNo
            XCheck = np.delete(XCheck, check)

        print(f"{len(XCheck_False)} out of {XCheckIn.shape[0]} frames from cluster {idx} will be dismissed")
        if XCheck.shape[0] < len(XCheck_False):
            print(f"For cluster {idx} more frames are dismissed than kept")

        # Übergabe: processing frames
        metric_Check1 = list()
        YCheck = YCheckIn[XCheck, :]
        mean_wfg = np.mean(YCheck, axis=0)
        WaveRef = crossval(mean_wfg, mean_wfg)
        for idy, value in enumerate(XCheck):
            WaveIn = crossval(YCheck[idy, :], mean_wfg)
            metric_Check1.append(calc_metric(WaveIn, WaveRef))
        data_process_XCheck[idx] = np.column_stack((XCheckIn[XCheck], idx + np.ones(len(XCheckIn[XCheck]))))
        data_process_YCheck[idx] = YCheckIn[XCheck, :]
        data_process_mean[idx] = mean_wfg
        data_process_metric[idx] = metric_Check1

        # Übergabe: dismissed frames
        metric_Check1 = []
        YCheck = YCheckIn[XCheck_False, :]
        mean_wfg = np.mean(YCheck, axis=0, dtype=np.float64)
        WaveRef = crossval(mean_wfg, mean_wfg)
        for idy, frame_idx in enumerate(XCheck_False):
            WaveIn = crossval(YCheck[idy, :], mean_wfg)
            metric_Check1.append(calc_metric(WaveIn, WaveRef))
        data_dismiss_XCheck[idx] = np.column_stack((XCheckIn[XCheck_False], idx + np.ones(len(XCheckIn[XCheck_False]))))
        data_dismiss_YCheck[idx] = YCheckIn[XCheck_False, :]
        data_dismiss_mean[idx] = mean_wfg
        data_dismiss_metric[idx] = metric_Check1

    del idx, idy, do_run, check, YCheck, YCheckIn, XCheckIn, XCheck, XCheck_False, WaveIn, WaveRef, IteNo, \
        metric_Check1, criteria
    print(" ... End of step #1")
    # endregion

    # region Processing: Merging Cluster
    data_2merge_XCheck = defaultdict()
    data_2merge_YCheck = defaultdict()
    data_2merge_mean = defaultdict()
    data_2merge_metric = defaultdict()
    data_2wrong_Xnew = defaultdict()
    data_2wrong_Ynew = defaultdict()
    data_2wrong_mean = defaultdict()

    data_2merge_XCheck[0] = data_process_XCheck[0]  # data_2merge[2]
    data_2merge_YCheck[0] = data_process_YCheck[0]  # data_2merge_[3]
    data_2merge_mean[0] = data_process_mean[0]  # data_2merge[4]

    data_2merge_number = 0
    data_2wrong_number = 0
    data_missed_new_XCheck = defaultdict()
    data_missed_new_YCheck = defaultdict()

    for idx in range(1, len(input_cluster)):
        Yraw_New = data_process_YCheck[idx]
        Ymean_New = data_process_mean[idx]
        Xraw_New = data_process_XCheck[idx]

        # Erste Prüfung: Mean-Waveform vergleichen mit bereits gemergten Clustern
        metric_Run0 = [calc_metric(crossval(Ymean_New, Ycheck_Mean), crossval(Ycheck_Mean, Ycheck_Mean)) for Ycheck_Mean
                       in data_2merge_mean.values()]

        # Entscheidung treffen
        candY = np.max(np.array(metric_Run0)[:, 4])
        candX = np.argmax(np.array(metric_Run0)[:, 4])
        if np.isnan(candX):
            # Keine Lösung vorhanden: Anhängen
            data_2wrong_number += 1
            data_2wrong_Xnew[idx] = Xraw_New
            data_2wrong_Ynew[idx] = Yraw_New
            data_2wrong_mean[idx] = Ymean_New
        elif metric_Run0[candX][4] >= criterion_Run0:
            # Zweite Prüfung: Einzel-Waveform mit Mean
            YCheck = np.vstack([data_2merge_YCheck[candX], Yraw_New])
            XCheck = np.vstack([data_2merge_XCheck[candX], Xraw_New])
            YMean = data_2merge_mean[candX]
            WaveRef = crossval(YMean, YMean)
            metric_Run1 = np.array([calc_metric(crossval(Y, YMean), WaveRef) for Y in YCheck])
            selOut = np.where(metric_Run1[:, 4] <= 0.92)[0]
            if selOut.size != 0:
                data_missed_new_XCheck[len(data_dismiss_XCheck) + 1] = np.vstack(
                    [data_missed_new_XCheck[idx], XCheck[selOut, :]])
                data_missed_new_YCheck[len(data_dismiss_YCheck) + 1] = np.vstack(
                    [data_missed_new_YCheck[idx], YCheck[selOut, :]])

                XCheck = np.delete(XCheck, selOut, axis=0)
                YCheck = np.delete(YCheck, selOut, axis=0)

            # Potentieller Match
            data_2merge_XCheck[candX] = XCheck
            data_2merge_YCheck[candX] = YCheck
            data_2merge_mean[candX] = np.mean(YCheck, axis=0, dtype=np.float64)
        else:
            # Neues Cluster
            data_2merge_number += 1
            data_2merge_XCheck[data_2merge_number] = Xraw_New
            data_2merge_YCheck[data_2merge_number] = Yraw_New
            data_2merge_mean[data_2merge_number] = Ymean_New
    print("End of step #2")

    data_dismiss_XCheck[len(data_dismiss_XCheck)] = data_missed_new_XCheck.get(len(data_dismiss_XCheck) + 1,
                                                                               np.array([]))
    data_dismiss_YCheck[len(data_dismiss_YCheck)] = data_missed_new_YCheck.get(len(data_dismiss_YCheck) + 1,
                                                                               np.array([]))
    data_dismiss_mean[len(data_dismiss_mean)] = np.mean(
        data_missed_new_YCheck.get(len(data_dismiss_mean) + 1, np.array([])), axis=0, dtype=np.float64)

    for idy, Yraw in data_2merge_YCheck.items():
        WaveRef = crossval(data_2merge_mean[idy], data_2merge_mean[idy])
        data_2merge_metric[idy] = [calc_metric(crossval(Y, data_2merge_mean[idy]), WaveRef) for Y in Yraw]

    del idx, idy, candX, candY, Yraw_New, Ymean_New, Xraw_New, WaveRef, Y, \
        data_missed_new_YCheck, data_missed_new_XCheck
    # endregion

    # region Post-Processing: Resorting dismissed frames
    data_restored = 0
    if setOptions['do_resort']:
        for idz, value in enumerate(data_dismiss_XCheck):
            pos_sel = data_dismiss_XCheck[value]
            frames_sel = data_dismiss_YCheck[value]

            for idx in range(frames_sel.shape[0]):
                metric_Run2 = np.array([calc_metric(crossval(frames_sel[idx, :], data_2merge_mean[idx2]),
                                                    crossval(data_2merge_mean[idx2], data_2merge_mean[idx2]))
                                        for idx2 in range(data_2merge_number)])
                # Decision
                selY, selX = np.max(metric_Run2[:, 4]), np.argmax(metric_Run2[:, 4])
                if selY >= criterion_Resort:
                    data_2merge_XCheck[selX] = np.vstack([data_2merge_XCheck[selX], pos_sel[idx, :]])
                    data_2merge_YCheck[selX] = np.vstack([data_2merge_YCheck[selX], frames_sel[idx, :]])
                    data_restored += 1
        print(" ... resorting dismissed frames to original")
    del idz, value, pos_sel, frames_sel, idx, metric_Run2, selY, selX
    # endregion

    # region Preparing: Transfer to new file
    output = {'frames': np.empty((0, 32), dtype=np.float64), 'cluster': np.empty((0,), dtype=np.int16)}
    data_process_num = 0

    for idx, value in enumerate(data_2merge_XCheck):
        X = np.array(value, dtype=np.int16) * np.ones(len(data_2merge_XCheck[value]), dtype=np.int16)
        Z = data_2merge_YCheck[value]

        output['cluster'] = np.concatenate([output['cluster'], X])
        output['frames'] = np.concatenate([output['frames'], Z])

        data_process_num += len(X)
    del idx, value
    # endregion

    # region Saving Output
    data_ratio_merged = data_process_num / frames_in_number
    data_ratio_dismiss = 1 - data_ratio_merged
    print(f"Percentage of overall kept frames: {data_ratio_merged * 100:.2f}")
    print(f"Percentage of overall dismissed frames: {data_ratio_dismiss * 100:.2f}")

    outdict = {'frames_in': output['frames'],
               'frames_cluster': output['cluster'],
               'data_ratio_merged': data_ratio_merged}

    savemat(setOptions['path2save'], outdict, appendmat=False, oned_as='column')
    print(" ... merged output generated")
    # endregion

    # region Plotting Output
    plot_results(data_2merge_XCheck, data_2merge_YCheck, data_2merge_mean, setOptions['path2fig'], '_ResultsMerged_Fig')
    plot_results(data_dismiss_XCheck, data_dismiss_YCheck, data_dismiss_mean, setOptions['path2fig'],
                 '_ResultsDismiss_Fig')
    # endregion

    print("This is the End!")


merge_datasets()
