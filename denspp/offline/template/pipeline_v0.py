import numpy as np
from os.path import abspath
from denspp.offline.pipeline import PipelineCMD
from denspp.offline.analog import (
    PreAmp, SettingsAMP,
    SettingsADC, SuccessiveApproximation as ADC0
)
from denspp.offline.preprocessing import (
    Filtering, SettingsFilter,
    SpikeDetection, SettingsSDA
)
from denspp.offline.ml import (
    FeatureExtraction, SettingsFeature,
    Clustering, SettingsCluster
)
from denspp.offline.postprocessing import (
    calc_spiketicks
)
from .pipeline_plot import plot_frames_feature, plot_transient_highlight_spikes, plot_transient_input_spikes


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
        self.SettingsDSP_LFP = SettingsFilter(
            gain=1,
            fs=fs_dig,
            n_order=2, f_filt=[0.1, 100],
            type='iir', f_type='butter', b_type='bandpass'
        )
        self.SettingsDSP_SPK = SettingsFilter(
            gain=1,
            fs=fs_dig,
            n_order=2, f_filt=[200, 8e3],
            type='iir', f_type='butter', b_type='bandpass'
        )
        # --- Options for Spike Detection and Frame Aligning
        self.SettingsSDA = SettingsSDA(
            fs=fs_dig, dx_sda=[1],
            mode_align=1,
            t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
            dt_offset=[0.1e-3, 0.1e-3],
            t_dly=0.4e-3,
            window_size=7,
            thr_gain=1.0,
            thr_min_value=100.0
        )
        # --- Options for MachineLearning Part
        self.SettingsFE = SettingsFeature()
        self.SettingsCL = SettingsCluster(
            type="kMeans",
            no_cluster=3
        )


class PipelineV0(PipelineCMD):
    def __init__(self, fs_ana: float, addon: str='_app') -> None:
        """Processing Pipeline for analysing transient data
        :param fs_ana:  Sampling rate of the input signal [Hz]
        :param addon:   String text with folder addon for generating result folder in runs
        """
        super().__init__()
        self._path2pipe = abspath(__file__)
        self.generate_run_folder('runs', addon)

        settings = SettingsPipe(
            bit_adc=12,
            adc_dvref=0.1,
            fs_ana=fs_ana,
            fs_dig=fs_ana,
            vss=-0.6,
            vdd=0.6,
        )
        self.__preamp0 = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__dsp0 = Filtering(settings.SettingsDSP_LFP)
        self.__dsp1 = Filtering(settings.SettingsDSP_SPK)
        self.__sda = SpikeDetection(settings.SettingsSDA)
        self.__fe = FeatureExtraction(settings.SettingsFE)
        self.__cl = Clustering(settings.SettingsCL)

    def do_plotting(self, data: dict, channel: int) -> None:
        """Function to plot results after processing
        :param data:        Dictionary with data content
        :param channel:     Integer of channel number
        """
        plot_transient_input_spikes(data, channel, path=self.path2save)
        plot_frames_feature(data, channel, path=self.path2save, take_feat_dim=[0, 1])
        plot_transient_highlight_spikes(data, channel, path=self.path2save, show_plot=True)

    def run_preprocessor(self, u_in: np.ndarray, spike_xpos: list=(), spike_xoffset: int=0) -> dict:
        """Function with methods for emulating pre-processor of the use-case-specific signal processor
        :param u_in:            Input signal
        :param spike_xpos:      List of all spike positions from groundtruth
        :param spike_xoffset:   Time delay between spike_xpos and real spike
        :return:                Dictionary with pre-processing results
        """
        # ---- Analogue Front End Module ----
        u_pre = self.__preamp0.pre_amp_chopper(u_in, np.array(self.__preamp0.vcm))['out']
        x_adc, _, u_quant = self.__adc.adc_ideal(u_pre)
        # ---- Digital Pre-processing ----
        x_lfp = self.__dsp0.filter(x_adc)
        x_spk = self.__dsp1.filter(x_adc)
        # ---- Spike detection incl. thresholding ----
        if len(spike_xpos):
            frames_align, frames_xpos = self.__sda.frame_generation_pos(
                x_adc, np.array(spike_xpos), spike_xoffset
            )[1:]
        else:
            x_dly = self.__sda.time_delay(x_spk)
            x_sda = self.__sda.sda_spb(x_spk, [200, 2e3])
            x_thr = self.__sda.thres_blackrock(x_sda)
            frames_orig, frames_align = self.__sda.frame_generation(
                x_dly, x_sda, x_thr
            )
        return {
            "fs_adc": self.__adc._settings.fs_adc,
            "fs_dig": self.__adc._settings.fs_dig,
            "u_in": u_in,
            "x_spk": np.array(x_spk, dtype=np.int16),
            "frames": frames_align
        }

    def run_classifier(self, data: dict) -> dict:
        """Function with methods for emulating classifier of the use-case-specific signal processor
        :param data:    Dictionary with pre-processed data / results
        :return:        Dictionary with pre-processing and classification results
        """
        if data['frames'][1].size == 0:
            features = np.zeros((1, ))
            data['frames'][2] = np.zeros((1, ))
        else:
            features = self.__fe.pca(data['frames'][0], num_features=3)
            data['frames'][2] = self.__cl.init(features)

        data["features"] = features
        return data

    def run_postprocessing(self, data: dict) -> dict:
        """Function with methods for post-processing the classified spike frames
        :param data:    Dictionary with pre-processed data / results
        :return:        Dictionary with pre-processing, classification and post-processed results
        """
        if data['frames'][1].size == 0:
            spike_ticks = np.zeros((1,))
        else:
            spike_ticks = calc_spiketicks(data['frames'][1])

        data["spike_ticks"] = spike_ticks
        return data

    def run(self, uin: np.ndarray) -> dict:
        """Function with methods for emulating the end-to-end signal processing for choicen use-case
        :param uin:     Input signal
        :return:        Dictionary with results
        """
        data = self.run_preprocessor(u_in=uin)
        data = self.run_classifier(data)
        return self.run_postprocessing(data)
