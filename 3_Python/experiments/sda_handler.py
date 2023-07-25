import numpy as np
import matplotlib.pyplot as plt

from sda_pipeline import Settings
from sda_pipeline import Pipeline_Digital as sda_pipeline
from src.signal_gen import EasyNeuralData_Generator
from src.metric import calculate_snr
from src.plotting import results_afe1

def characterize_sda_output(spk_pos_in: np.ndarray, spk_pos_sda: np.ndarray, fs: float) -> [float, float, float]:
    true_right_cnt = 0
    false_right_cnt = 0
    time_range = int(1.2e-3 * fs)
    dt_range = int(1.2e-3 * fs)

    val_old = 0
    for val_in in spk_pos_sda:
        detected = False
        for val_ref in spk_pos_in:
            if np.abs(val_in - val_ref) <= time_range and not detected and np.abs(val_in - val_old) >= dt_range:
                val_old = val_in
                true_right_cnt += 1
                detected = True

        if not detected:
            false_right_cnt += 1

    dt_acc = float(true_right_cnt / spk_pos_sda.size)
    true_right_rate = float(true_right_cnt / spk_pos_sda.size)
    false_right_rate = float(false_right_cnt / spk_pos_sda.size)

    return dt_acc, true_right_rate, false_right_rate

def plot_results_single(time: np.ndarray, spk_signal: np.ndarray, spk_frm_in: np.ndarray, methods: str) -> None:
    spk_mean_in = np.mean(spk_frm_in, axis=0)
    #spk_mean_filt = np.mean(spk_frm_filt, axis=0)

    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    ax1.set_title(methods)
    ax1.plot(time, 1e6 * spk_signal, color='k')
    ax1.set_ylabel('U_elec [µV]')
    ax1.set_xlabel('Time [s]')

    ax2.plot(1e6 * np.transpose(spk_frm_in), color='k')
    ax2.plot(1e6 * spk_mean_in, color='r')
    ax2.set_ylabel('U_elec [µV]')
    ax2.set_xlabel('Position')

    plt.tight_layout()

def plot_results_sweep(spk_firing_rate: np.ndarray, snr: np.ndarray, dt_acc: np.ndarray, tr_rate: np.ndarray, fr_rate: np.ndarray, methods: str) -> None:
    plt.figure()
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)

    text_spk = list()
    for val in spk_firing_rate:
        text_spk.append("FR = " + str(val) + " Hz")


    dcolor = [0.42, 0.5, 0.58, 0.66, 1.0]
    for idx0, val in enumerate(spk_firing_rate):
        ax1.plot(snr, dt_acc[idx0, :], color='k', marker='.', alpha=dcolor[idx0])
        ax2.plot(snr, tr_rate[idx0, :], color='r', marker='.', alpha=dcolor[idx0])
        ax2.plot(snr, fr_rate[idx0, :], color='b', marker='.', alpha=dcolor[idx0])

    ax1.set_title(methods)
    ax1.set_ylabel("Acc. [%]")
    ax1.legend(text_spk, loc="lower right")
    ax1.grid()

    ax2.set_ylabel("Rate [%]")
    ax2.set_xlabel("SNR [dB]")
    ax2.legend(["True Positive", "False Positive"], loc="right")
    ax2.grid()

    plt.tight_layout()

# --------------- MAIN PROGRAMME
if __name__ == "__main__":
    # --- Configuration of SDA properties
    mode_sda = 3
    mode_thr = 1

    # --- Configuration of neural input
    no_spk = 200
    spk_amplitude = 100e-6
    fs = 25e3
    spk_firing_rate = [20]
    # np.linspace(20, 200, num=3, endpoint=True)
    snr_in = [0]
    # np.linspace(0, 6, num=11, endpoint=True)
    # --- Preparing the experiment
    settings_exp = Settings()
    settings_exp.SettingsAMP.fs_ana = fs
    settings_exp.SettingsADC.fs_ana = fs
    settings_exp.SettingsADC.fs_dig = fs
    handler_exp = sda_pipeline(settings_exp, fs)

    # --- Loading the input raw data
    t_frame = settings_exp.SettingsSDA.t_frame_lgth
    data = EasyNeuralData_Generator(no_spk, fs)

    dt_acc = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))
    tr_rate = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))
    fr_rate = np.zeros(shape=(len(spk_firing_rate), len(snr_in)))

    for idx0, spk_fr in enumerate(spk_firing_rate):
        print(f"\nStart Test at firing rate of {spk_fr: .2f} Hz:")
        for idx1, snr_run in enumerate(snr_in):
            spk_signal, spk_pos = data.gen_spike_activity(spk_amplitude, t_frame, spk_fr, snr_run)
            time = data.time
            spk_frames_in = data.cut_frames(spk_signal, spk_pos, t_frame)
            spk_mean = np.mean(spk_frames_in, axis=0)

            # --- Process the data
            handler_exp.run_preprocess(spk_signal, mode_sda, mode_thr)
            spk_frames_out = handler_exp.frames_orig
            spk_frames_pos = handler_exp.x_pos[:, 0]
            methods = handler_exp.used_methods

            dt_acc[idx0, idx1], tr_rate[idx0, idx1], fr_rate[idx0, idx1] = characterize_sda_output(spk_pos, spk_frames_pos, fs)
            print(f"... @SNR = {snr_run: .1f} dB --> Acc. = {100 * dt_acc[idx0, idx1]:.2f} %, TR = {100 * tr_rate[idx0, idx1]: .2f} %, FR = {100 * fr_rate[idx0, idx1]: .2f} %")

            if len(snr_in) == 1 and len(spk_firing_rate) == 1:
                # --- Post-Processing and SNR calculation (single trial)
                spk_snr_in = np.zeros(shape=(spk_frames_in.shape[0],))
                for idx, frame in enumerate(spk_frames_in[:, ]):
                    spk_snr_in[idx] = calculate_snr(frame, spk_mean)

                spk_snr_out = np.zeros(shape=(spk_frames_in.shape[0],))
                for idx, frame in enumerate(spk_frames_in[:, ]):
                    spk_snr_out[idx] = calculate_snr(frame, spk_mean)

                print(f"Results: SNR_in = {np.mean(spk_snr_in): .3f} dB +/- {np.std(spk_snr_in): .3f}")
                print(f"Results: SNR_out = {np.mean(spk_snr_out): .3f} dB +/- {np.std(spk_snr_out): .3f}")
                results_afe1(handler_exp, "", 0)
                plot_results_single(time, spk_signal, spk_frames_in, methods)

    # --- Plotting for sweep
    if len(snr_in) != 1 or len(spk_firing_rate) != 1:
        plot_results_sweep(spk_firing_rate, snr_in, dt_acc, tr_rate, fr_rate, methods)

    plt.show()
