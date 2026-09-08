from copy import deepcopy
from tqdm import tqdm
from shutil import rmtree

from dataclasses import dataclass
from tempfile import TemporaryDirectory
from enum import Enum
from pathlib import Path
import numpy as np

from matplotlib import pyplot as plt
from elasticai.preprocessor.waveform import WaveformGenerator

from denspp.offline import get_path_to_project
from denspp.offline.preprocessing import do_fft
from denspp.offline.metric import calculate_snr, calculate_cosine_similarity, calculate_error_mae, calculate_error_mbe
from denspp.offline.metric.electrical import calculate_total_harmonics_distortion_from_transient, calculate_enob_from_transient, calculate_sinad_from_transient
from denspp.offline.plot_helper import save_figure


class TargetCreateDesign(Enum):
    FPGA = "fpga"
    MCU = "mcu"
    ASIC = "asic"
    PC = "pc"


@dataclass
class Signals:
    target_freq: float
    sampling_rate: float
    timestamps: np.ndarray
    signal: np.ndarray


@dataclass
class Metrics:
    MAE: float
    MBE: float
    CosineSimilarity: float
    OriginalTHD: float
    WaveformTHD: float
    OriginalSNR: float
    WaveformSNR: float
    OriginalSINAD: float
    WaveformSINAD: float
    OriginalENOB: float
    WaveformENOB: float


@dataclass
class SettingsAnalysis:
    method: str
    t_end: float
    target_freq: float
    sampling_rate: float
    num_params: int
    is_signed: bool
    target: TargetCreateDesign
    bitwidth: int
    do_opt: bool


def reconstruct_full_signal(quarterwave: list[int], without_duplicates: bool=True) -> list[int]:
    q2 = list(quarterwave)
    q1 = q2[::-1]
    q3 = [-v for v in q1]
    q4 = [-v for v in q2]
    if without_duplicates:
        return q1[:-1] + q2[:-1] + q3[:-1] + q4
    else:
        return q1 + q2 + q3 + q4


def generate_noise(num_samples: int) -> np.ndarray:
    return 1 * np.random.randn(num_samples).astype(float)


def generate_transient_signal(
    id: str, settings: SettingsAnalysis,
) -> Signals:
    full_waveform = get_hardware_waveform(
        id=id,
        settings=settings,
    )
    num_samples = int(settings.t_end * settings.sampling_rate)
    num_interleaved = int(settings.sampling_rate / settings.target_freq / (len(full_waveform)-1))

    timestamps = np.linspace(start=0, stop=settings.t_end, num=num_samples, endpoint=True)
    signal = np.zeros_like(timestamps)

    start_idx = 0
    stop_idx = 0
    idx = 0
    waveform_used = full_waveform[:-1]
    num_waveform = len(waveform_used)
    while stop_idx <= num_samples:
        if stop_idx == 0:
            start_idx = 0
            stop_idx = start_idx + int(num_interleaved/2)
        else:
            start_idx = stop_idx
            stop_idx = start_idx + num_interleaved
        signal[start_idx:stop_idx] = waveform_used[idx % num_waveform]
        idx += 1

    return Signals(
        target_freq=settings.target_freq,
        timestamps=timestamps,
        signal=signal,
        sampling_rate=timestamps.size / settings.t_end,
    )


def generate_reference_signal(
        settings: SettingsAnalysis,
) -> Signals:
    offset = 0 if settings.is_signed else 2 ** (settings.bitwidth-1)
    amplitude = 2 ** (settings.bitwidth-1)
    num_samples = int(settings.t_end * settings.sampling_rate)

    time = np.linspace(start=0, stop=settings.t_end, num=num_samples, endpoint=True)
    return Signals(
        target_freq=settings.target_freq,
        timestamps=time,
        signal=amplitude * np.sin(2 * np.pi * settings.target_freq * time) + offset,
        sampling_rate=settings.sampling_rate
    )


def get_hardware_waveform(id: str, settings: SettingsAnalysis) -> list[int]:
    with TemporaryDirectory() as tmpdir:
        data = WaveformGenerator(
            sampling_rate=settings.sampling_rate,
        ).create_design(
            waveform=settings.method,
            num_params=settings.num_params if not settings.do_opt else int(settings.num_params / 4),
            is_signed=settings.is_signed,
            target=settings.target.value,
            bitwidth=settings.bitwidth,
            id=id,
            path2save=Path(tmpdir),
            use_bram=False,
            do_opt=settings.do_opt,
        )

        if not settings.do_opt:
            return data[::-1]
        else:
            return reconstruct_full_signal(data, without_duplicates=True)


def plot_transient_results(signal: Signals, reference: Signals, path2save: Path, id: str, do_show: bool=False, do_save: bool=False) -> Metrics:
    noise_ref = generate_noise(num_samples=reference.signal.size)
    noise_sig = generate_noise(num_samples=signal.signal.size)
    signal_ref = reference.signal + noise_ref
    signal_sig = signal.signal + noise_sig

    if do_show or do_save:
        num_periods_idx = [int(signal.sampling_rate / signal.target_freq), int(reference.sampling_rate / reference.target_freq)]
        start_idx = num_periods_idx
        stop_idx = [start + 3 * period for start, period in zip(start_idx, num_periods_idx)]
        # --- Transient signal
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False)
        axs[0].plot(signal.timestamps[start_idx[0]:stop_idx[0]], signal_sig[start_idx[0]:stop_idx[0]], color="k", marker=".", markersize=12, label="waveform")
        axs[0].plot(reference.timestamps[start_idx[1]:stop_idx[1]], signal_ref[start_idx[1]:stop_idx[1]], color="r", label="reference")
        axs[0].set_xlim([signal.timestamps[start_idx[0]], signal.timestamps[stop_idx[0]]])
        axs[0].set_xlabel("Time / s", size=12)
        axs[0].set_ylabel("Signal s(t)", size=12)

        # --- FFT
        window_method = "hamming"
        f_sig, y_sig = do_fft(y=signal_sig, fs=signal.sampling_rate, method_window=window_method)
        axs[1].loglog(f_sig, y_sig, color="k", marker=".", markersize=12, label="waveform")

        f_ref, y_ref = do_fft(y=signal_ref, fs=reference.sampling_rate, method_window=window_method)
        axs[1].loglog(f_ref, y_ref, color="r", marker=".", markersize=12, label="reference")
        axs[1].set_xlim([f_ref[0], reference.sampling_rate])
        axs[1].set_xlabel("Frequency / Hz", size=12)
        axs[1].set_ylabel("Signal s(f)", size=12)

        for ax in axs:
            ax.grid(True)
            ax.legend(loc="upper left")
        plt.tight_layout()
        if do_save:
            save_figure(fig, path=(path2save / "transient").as_posix(), name=f"transient_{id}", formats=["jpg"])
        if do_show:
            plt.show(block=True)

    return Metrics(
        MAE=calculate_error_mae(y_pred=signal.signal, y_true=reference.signal),
        MBE=calculate_error_mbe(y_pred=signal.signal, y_true=reference.signal),
        CosineSimilarity=calculate_cosine_similarity(y_pred=signal.signal, y_true=reference.signal),
        OriginalTHD=calculate_total_harmonics_distortion_from_transient(signal=signal_ref, fs=reference.sampling_rate),
        WaveformTHD=calculate_total_harmonics_distortion_from_transient(signal=signal_sig, fs=signal.sampling_rate),
        OriginalSNR=calculate_snr(data=reference.signal, mean=noise_ref),
        WaveformSNR=calculate_snr(data=signal.signal, mean=noise_sig),
        OriginalSINAD=calculate_sinad_from_transient(signal=signal_ref, fs=signal.sampling_rate, num_harmonics=1),
        WaveformSINAD=calculate_sinad_from_transient(signal=signal_sig, fs=reference.sampling_rate, num_harmonics=1),
        OriginalENOB=calculate_enob_from_transient(signal=signal_ref, fs=reference.sampling_rate, num_harmonics=1),
        WaveformENOB=calculate_enob_from_transient(signal=signal_sig, fs=reference.sampling_rate, num_harmonics=1),
    )


def plot_metric_results(target_params: np.ndarray, target_name: str, metric: list[Metrics], path2save: Path, do_show: bool=True) -> None:
    data_thd_original = np.asarray([val.OriginalTHD for val in metric])
    data_thd_waveform = np.asarray([val.WaveformTHD for val in metric])
    data_snr_original = np.asarray([val.OriginalSNR for val in metric])
    data_snr_waveform = np.asarray([val.WaveformSNR for val in metric])
    data_cosine_similarity = np.asarray([val.CosineSimilarity for val in metric])
    data_mbe = np.asarray([val.MBE for val in metric])

    fig, ax1 = plt.subplots()
    ax1.semilogx(target_params, data_mbe, color="k", marker=".", markersize=12, label="MAE")
    ax1.semilogx(target_params, data_cosine_similarity, color="g", marker=".", markersize=12, label="Cosine Similarity")
    ax1.set_ylabel("Metrics]", size=12)
    ax1.set_xlabel(target_name, size=12)

    ax2 = ax1.twinx()
    ax2.semilogx(target_params, data_thd_original, color="r", marker="x", markersize=8, label="THD (Original) [dB]")
    ax2.semilogx(target_params, data_thd_waveform, color="r", marker="h", markersize=8, label="THD (Waveform) [dB]")
    ax2.semilogx(target_params, data_snr_original, color="b", marker="x", markersize=8, label="SNR (Original) [dB]")
    ax2.semilogx(target_params, data_snr_waveform, color="b", marker="h", markersize=8, label="SNR (Waveform) [dB]")
    ax2.set_ylabel("Metrics [dB]", size=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    plt.grid()
    plt.tight_layout()
    save_figure(fig, path=path2save.as_posix(), name=f"metric")
    if do_show:
        plt.show()


if __name__ == "__main__":
    build_plots = False
    show_plots = False

    path2save = get_path_to_project() / "runs" / "waveform"
    if path2save.exists():
        rmtree(path2save)
    path2save.mkdir(parents=True, exist_ok=True)

    sets_start = SettingsAnalysis(
        method="SINE_FULL",
        t_end=120.0,
        target_freq=1.0,
        sampling_rate=1e5,
        num_params=21,
        is_signed=True,
        target=TargetCreateDesign.FPGA,
        bitwidth=12,
        do_opt=False,
    )

    results = list()
    param_range = np.logspace(start=-1, stop=3, num=17, endpoint=True)
    pbar = tqdm(param_range)
    for idx, target_freq in enumerate(pbar):
        pbar.set_description(f"Used target frequency: {target_freq:.2f} Hz")
        sets_used: SettingsAnalysis = deepcopy(sets_start)
        sets_used.target_freq = target_freq

        reconstruct = generate_transient_signal(id="0", settings=sets_used)
        reference = generate_reference_signal(settings=sets_used)
        metrics = plot_transient_results(
            signal=reconstruct,
            reference=reference,
            id=f"{idx:02d}",
            path2save=path2save,
            do_show=show_plots,
            do_save=build_plots,
        )
        results.append(metrics)
        pbar.set_postfix(
            mae=f"{metrics.MAE:.2f}",
            mbe=f"{metrics.MBE:.2f}",
            similarity=f"{100 * metrics.CosineSimilarity:.2f} %",
            thd=f"{metrics.WaveformTHD:.2f} dB",
            snr=f"{metrics.WaveformSNR:.2f} dB",
            sinad=f"{metrics.WaveformSINAD:.2f} dB",
            enob=f"{metrics.WaveformENOB:.2f} bit",
        )

    plot_metric_results(
        target_params=param_range,
        target_name="Target Frequency / Hz",
        metric=results,
        path2save=path2save,
    )
