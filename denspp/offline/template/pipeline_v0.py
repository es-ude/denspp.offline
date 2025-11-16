import numpy as np
from os.path import abspath
from denspp.offline.pipeline import PipelineCMD
from denspp.offline.analog import (
    PreAmp, SettingsAMP,
    SettingsADC, SuccessiveApproximation as ADC0
)
from denspp.offline.preprocessing import (
    Filtering, SettingsFilter,
    SpikeDetection, SettingsSDA, FrameWaveform
)
from denspp.offline.ml import (
    FeatureExtraction, SettingsFeature,
    Clustering, SettingsCluster
)
from denspp.offline.postprocessing import (
    calc_spike_ticks
)


class SettingsPipe:
    def _overwrite_power_supply(self, vss: float, vdd: float) -> None:
        a = [method for method in dir(self) if 'Settings' in method and '__' not in method]
        for setting in a:
            set0 = getattr(self, setting)
            set0.vss = vss
            set0.vdd = vdd

    def __init__(self, bit_adc: int, adc_dvref: float,
                 fs_ana: float, fs_dig: float,
                 vss: float, vdd: float
                 ) -> None:
        """Settings class for setting-up the pipeline
        :param bit_adc:     Bit-resolution of used ADC
        :param adc_dvref:   Diff. voltage of ADC reference voltage [V]
        :param fs_ana:      Sampling frequency of Analog Input [Hz]
        :param fs_dig:      Sampling frequency of ADC output [Hz]
        :param vss:         Negative Supply Voltage [V]
        :param vdd:         Positive Supply Voltage [V]
        """
        self._overwrite_power_supply(vss, vdd)

        # --- Digital filtering for ADC output and CIC
        self.SettingsAMP = SettingsAMP(
            vss=-0.6, vdd=0.6,
            fs_ana=fs_ana,
            gain=40,
            n_filt=1, f_filt=[0.1, 8e3], f_type="band",
            offset=1e-6, noise_en=True,
            f_chop=10e3,
            noise_edev=100e-9
        )
        self.SettingsADC = SettingsADC(
            vdd=0.6, vss=-0.6,
            is_signed=True,
            dvref=adc_dvref,
            fs_ana=fs_ana,
            fs_dig=fs_dig, osr=1, Nadc=bit_adc
        )
        # --- Digital filtering for ADC output and CIC
        self.SettingsDSP_SPK = SettingsFilter(
            gain=1,
            fs=fs_dig,
            n_order=2, f_filt=[200, 8e3],
            type='iir', f_type='butter', b_type='bandpass'
        )
        # --- Options for Spike Detection and Frame Aligning
        self.SettingsSDA = SettingsSDA(
            sampling_rate=fs_dig, dx_sda=[2],
            mode_align='min',
            mode_thr='rms_black',
            mode_sda='neo',
            t_frame_length=1.6e-3,
            t_frame_start=0.4e-3,
            dt_offset=0.1e-3,
            thr_gain=1.25
        )
        # --- Options for MachineLearning Part
        self.SettingsFE = SettingsFeature()
        self.SettingsCL = SettingsCluster(
            type="kMeans",
            no_cluster=3
        )


class PipelineV0(PipelineCMD):
    def __init__(self, fs_ana: float, build_folder: bool=True) -> None:
        """Processing Pipeline for analysing transient data
        :param fs_ana:          Sampling rate of the input signal [Hz]
        :param build_folder:    Boolean for building the report folder and build report
        """
        super().__init__()
        self.fs_ana = fs_ana
        self.fs_dig = fs_ana

        self._path2pipe = abspath(__file__)
        if build_folder:
            self.generate_run_folder()

        settings = SettingsPipe(
            bit_adc=12,
            adc_dvref=0.1,
            fs_ana=self.fs_ana,
            fs_dig=self.fs_dig,
            vss=-0.6,
            vdd=0.6,
        )
        self.__amp = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__dsp = Filtering(settings.SettingsDSP_SPK)
        self.__sda = SpikeDetection(settings.SettingsSDA)
        self.__fe = FeatureExtraction(settings.SettingsFE)
        self.__cl = Clustering(settings.SettingsCL)

    def update_sampling_rate(self, fs_analog: float, fs_digital: float) -> None:
        self.fs_ana = fs_analog
        self.fs_dig = fs_digital

    def do_plotting(self, data: dict, channel: int) -> None:
        """Function to plot results after processing
        :param data:        Dictionary with data content
        :param channel:     Integer of channel number
        """
        from .pipeline_plot import plot_frames_feature, plot_transient_highlight_spikes, plot_transient_input_spikes
        plot_transient_input_spikes(data, channel, path=self.path2save)
        plot_frames_feature(data, channel, path=self.path2save, take_feat_dim=[0, 1])
        plot_transient_highlight_spikes(data, channel, path=self.path2save, show_plot=True)

    def run_preprocessor(self, u_in: np.ndarray, frames_xpos: list=(), frames_xoff: float=0.) -> dict:
        """Function with methods for emulating pre-processor of the use-case-specific signal processor
        :param u_in:        Input signal
        :param frames_xpos: List of all spike positions from ground-truth
        :param frames_xoff: Time delay between spike_xpos and real spike
        :return:            Dictionary with pre-processing results
        """
        # ---- Analogue Front End Module ----
        u_pre = self.__amp.pre_amp_chopper(u_in, np.array(self.__amp.vcm))['out']
        x_adc = self.__adc.adc_ideal(u_pre)[0]
        # ---- Digital Pre-processing ----
        x_spk = self.__dsp.filter(x_adc)
        # ---- Spike detection incl. thresholding ----
        if len(frames_xpos):
            frames = self.__sda.get_spike_waveforms_from_positions(
                xraw=x_spk,
                xpos=np.array(frames_xpos),
                xoffset=int(frames_xoff* self.fs_dig)
            )
        else:
            frames = self.__sda.get_spike_waveforms(
                xraw=x_spk,
                do_abs=False
            )
        return {
            "fs_ana": self.fs_ana,
            "fs_dig": self.fs_dig,
            "u_in": u_in,
            "x_spk": np.array(x_spk, dtype=np.int16),
            "frames": frames
        }

    def run_classifier(self, data: dict) -> dict:
        """Function with methods for emulating classifier of the use-case-specific signal processor
        :param data:    Dictionary with pre-processed data / results
        :return:        Dictionary with pre-processing and classification results
        """
        frames: FrameWaveform = data['frames']
        if frames.num_samples == 0:
            features = np.zeros((1, ))
        else:
            features = self.__fe.pca(frames.waveform, num_features=3)
            frames.label = self.__cl.init(features)

        data["features"] = features
        return data

    @staticmethod
    def run_postprocessor(data: dict) -> dict:
        """Function with methods for post-processing the classified spike frames
        :param data:    Dictionary with pre-processed data / results
        :return:        Dictionary with pre-processing, classification and post-processed results
        """
        frames: FrameWaveform = data['frames']
        if frames.waveform.size == 0:
            spike_ticks = np.zeros((1,))
        else:
            spike_ticks = calc_spike_ticks(frames)

        data["spike_ticks"] = spike_ticks
        return data

    def run(self, u_in: np.ndarray) -> dict:
        """Function with methods for emulating the end-to-end signal processing for selected use-case
        :param u_in:    Input signal
        :return:        Dictionary with results
        """
        data = self.run_preprocessor(u_in)
        data = self.run_classifier(data)
        return self.run_postprocessor(data)
