from os.path import abspath
import numpy as np
from denspp.offline.pipeline.pipeline_cmds import PipelineCMD
from denspp.offline.analog.amplifier.pre_amp import PreAmp, SettingsAMP
from denspp.offline.analog.adc import SettingsADC
from denspp.offline.analog.adc.adc_sar import SuccessiveApproximation as ADC0
from denspp.offline.preprocessing.sda import SpikeDetection, SettingsSDA


class SettingsPipe:
    def __init__(self, bit_adc: int=12, adc_dvref: float=0.1, fs_ana: float=20e3, fs_dig: float=20e3):
        """Settings class for setting-up the pipeline
            :param bit_adc:     Bit-resolution of used ADC
            :param adc_dvref:   Diff. voltage of ADC reference voltage [V]
            :param fs_ana:      Sampling frequency of Analog Input [Hz]
            :param fs_dig:      Sampling frequency of ADC output [Hz]
        """
        self.SettingsAMP = SettingsAMP(
            vss=-0.6, vdd=0.6,
            fs_ana=fs_ana,
            gain=100,
            n_filt=2, f_filt=[200, 8e3], f_type="band",
            offset=0e-6, noise_en=False,
            f_chop=20e3,
            noise_edev=100e-9
        )
        self.SettingsADC = SettingsADC(
            vdd=0.6, vss=-0.6,
            is_signed=True, dvref=adc_dvref,
            fs_ana=fs_ana,
            fs_dig=fs_dig, osr=1, Nadc=bit_adc
        )
        self.SettingsSDA = SettingsSDA(
            fs=fs_dig, dx_sda=[1],
            mode_align=2,
            t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
            dt_offset=[0.4e-3, 0.4e-3],
            t_dly=0.4e-3,
            window_size=7,
            thr_gain=1,
            thr_min_value=100
        )


class PipelineV0_Merge(PipelineCMD):
    def __init__(self, fs_ana: float, addon: str='_app'):
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
            fs_dig=fs_ana
        )
        self.fs_ana = fs_ana
        self.fs_adc = settings.SettingsADC.fs_adc
        self.fs_dig = settings.SettingsADC.fs_dig

        self.__preamp = PreAmp(settings.SettingsAMP)
        self.__adc = ADC0(settings.SettingsADC)
        self.__sda = SpikeDetection(settings.SettingsSDA)
        self.frame_left_windowsize = self.__sda.frame_start
        self.frame_right_windowsize = self.__sda.frame_ends

    def do_plotting(self, data: dict, channel: int) -> None:
        """Function to plot results after processing
        :param data:        Dictionary with data content
        :param channel:     Integer of channel number
        """
        pass

    def run(self, uin: np.ndarray, spike_xpos: list=(), spike_xoffset: int=0) -> dict:
        """Processing the raw data for frame generation
        :param uin:             Numpy Array of the 1D-transient signal
        :param spike_xpos:      List of all spike positions from groundtruth
        :param spike_xoffset:   Time delay between spike_xpos and real spike
        :return:                Dictionary with results
        """
        # --- Analogue Frontend
        u_pre = self.__preamp.pre_amp(uin, self.__preamp.vcm)
        x_adc = self.__adc.adc_ideal(u_pre)[0]

        # --- Frame Extraction
        frames_align, frames_xpos = self.__sda.frame_generation_pos(
            x_adc, np.array(spike_xpos), spike_xoffset
        )[1:]
        return {
            "fs_ana": self.fs_ana, "fs_adc": self.fs_adc, "fs_dig": self.fs_dig,
            "u_in": uin, "x_adc": x_adc, "frames_align": frames_align, "frames_xpos": frames_xpos
        }
