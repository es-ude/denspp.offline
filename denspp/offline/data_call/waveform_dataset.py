import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from .waveform_generator import WaveformGenerator
from denspp.offline.analog.dev_noise import SettingsNoise, RecommendedSettingsNoise


@dataclass
class SettingsWaveformDataset:
    """Settings Class for building the Waveform Dataset
    Attributes:
        wfg_type:       List with waveform type
        wfg_freq:       List with frequencies of each waveform
        num_samples:    Number of samples for each class
        time_idle:      Additional time window at the beginning and ending of each sample with zero values [in %]
        scale_amp:      Scaling factor for all amplitudes
        sampling_rate:  Sampling rate of waveforms
        noise_add:      Boolean for adding noise to waveforms
        noise_pwr_db:   Float
        do_normalize:   Boolean for normalizing the RMS of all waveforms to have same charge injection
    """
    wfg_type: list
    wfg_freq: list
    num_samples: int
    time_idle: float
    scale_amp: float
    sampling_rate: float
    noise_add: bool
    noise_pwr_db: float
    do_normalize: bool


DefaultSettingsWaveformDataset = SettingsWaveformDataset(
    wfg_type=['RECT_HALF', 'RECT_FULL', 'LIN_RISE', 'LIN_FALL', 'SINE_HALF', 'SINE_HALF_INV', 'SINE_FULL', 'TRI_HALF', 'TRI_FULL', 'SAW_POS', 'SAW_NEG', 'GAUSS'],
    wfg_freq=[1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2],
    num_samples=1000,
    time_idle=20,
    scale_amp=1.0,
    sampling_rate=20e3,
    noise_add=True,
    noise_pwr_db=-30.0,
    do_normalize=False
)


def build_waveform_dataset(settings_data: SettingsWaveformDataset, settings_noise: SettingsNoise=RecommendedSettingsNoise) -> dict:
    """Building a dataset of different waveform styles
    :param settings_data:   Class for generating the dataset
    :param settings_noise:  Dataclass for handling the noise behaviour
    :returns:               Returning a Dictionary with ['data', 'label', and 'dict']
    """
    assert len(settings_data.wfg_type) == len(settings_data.wfg_freq), "List have not the same length"
    settings0 = deepcopy(settings_noise)
    settings0.wgn_dB = settings_data.noise_pwr_db
    wfg_generator = WaveformGenerator(
        sampling_rate=settings_data.sampling_rate,
        add_noise=False,
        settings_noise=settings0
    )

    # --- Generation of signal
    num_class_samples = settings_data.num_samples
    num_total_samples = len(settings_data.wfg_type) * num_class_samples
    num_window = int((1 + 2 * settings_data.time_idle / 100) * settings_data.sampling_rate / min(settings_data.wfg_freq))
    time_point_min = settings_data.time_idle / 100 / min(settings_data.wfg_freq)

    waveforms_signals = np.zeros((num_total_samples, num_window), dtype=float)
    waveforms_classes = np.zeros((num_total_samples,), dtype=int)
    waveforms_rms = np.zeros(num_total_samples, )

    for idx, (sel_wfg, freq_wfg) in enumerate(zip(settings_data.wfg_type, settings_data.wfg_freq)):
        for num_ite in range(0, settings_data.num_samples):
            waveform = wfg_generator.generate_waveform(
                time_points=[time_point_min],
                time_duration=[1 / freq_wfg],
                waveform_select=[sel_wfg],
                polarity_cathodic=[False]
            )
            signal = waveform['sig'] if not settings_data.do_normalize else waveform['sig'] / waveform['rms']
            waveforms_signals[idx * num_class_samples + num_ite, :signal.size] = settings_data.scale_amp * signal
            waveforms_classes[idx * num_class_samples + num_ite] = idx
            waveforms_rms[idx * num_class_samples + num_ite] = waveform['rms']

    # --- Getting dictionary of signal type
    noise = wfg_generator.generate_noise(waveforms_signals.shape) if settings_data.noise_add else np.zeros_like(waveforms_signals)
    waveforms_dict = [type for type in settings_data.wfg_type if type in wfg_generator.get_dictionary_classes()]
    return {'data': waveforms_signals + noise, 'label': waveforms_classes, 'dict': waveforms_dict, 'fs': settings_data.sampling_rate}
