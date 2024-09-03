import numpy as np
from package.pipeline_cmds import PipelineSignal, PipelineCMD
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.adc_basic import SettingsADC
from package.analog.adc_sar import ADC_SAR as ADC0
from package.digital.dsp import DSP, SettingsDSP
from package.digital.sda import SpikeDetection, SettingsSDA


# --- Configuring the src_neuro
class _SettingsPipe:
    """Settings class for handling the src_neuro setting"""
    SettingsAMP = SettingsAMP(
        vss=-0.6, vdd=0.6,
        fs_ana=100,
        gain=40,
        n_filt=1, f_filt=[10, 8e3], f_type="band",
        offset=1e-6,
        noise_en=True,
        noise_edev=1e-9,
        f_chop=10e3
    )
    SettingsADC = SettingsADC(
        vdd=0.6, vss=-0.6,
        type_out="signed",
        dvref=0.1,
        fs_ana=100,
        fs_dig=20e3, osr=1, Nadc=12
    )

    # --- Digital filtering for ADC output and CIC
    SettingsDSP_SPK = SettingsDSP(
        gain=1,
        fs=SettingsADC.fs_adc,
        n_order=2, f_filt=[100, 8e3],
        type='iir', f_type='butter', b_type='band',
        t_dly=0
    )
    SettingsSDA = SettingsSDA(
        fs=SettingsADC.fs_adc, dx_sda=[2, 3, 4, 5, 6],
        mode_align=1,
        t_frame_lgth=1.6e-3, t_frame_start=0.4e-3,
        dt_offset=[0.1e-3, 0.1e-3],
        t_dly=0.4e-3,
        window_size=7,
        thr_gain=1
    )


class Pipeline_Digital(PipelineCMD):
    """Pipeline for processing SDA in digital approaches"""
    def __init__(self, fs: float):
        super().__init__()

        settings = _SettingsPipe()
        self.preamp0 = PreAmp(settings.SettingsAMP)
        self.adc = ADC0(settings.SettingsADC)
        self.dsp1 = DSP(settings.SettingsDSP_SPK)
        self.sda = SpikeDetection(settings.SettingsSDA)
        self.used_methods = None
        self.mode_sda = None
        self.mode_thr = None

        self.signals = PipelineSignal()
        self.signals.fs_ana = fs
        self.signals.fs_adc = settings.SettingsADC.fs_adc
        self.signals.fs_dig = settings.SettingsADC.fs_dig

    def define_sda(self, mode_sda: int, mode_thr: int) -> None:
        self.mode_sda = mode_sda
        self.mode_thr = mode_thr

    def run_preprocess(self, uin: np.ndarray, do_smooth=False, do_get_frames=False) -> None:
        self.signals.u_in = uin
        u_inn = np.array(self.preamp0.vcm)
        # ---- Analogue Front End Module ----
        self.signals.u_pre = self.preamp0.pre_amp(self.signals.u_in, u_inn)
        self.signals.x_adc = self.adc.adc_ideal(self.signals.u_pre)[0]
        # ---- Digital Pre-processing ----
        self.signals.x_spk = self.dsp1.filter(self.signals.x_adc)
        # ---- Spike detection incl. thresholding ----
        self.signals.x_dly = self.sda.time_delay(self.signals.x_spk)
        self.__do_sda(self.signals.x_spk, self.mode_sda, self.mode_thr, do_smooth=do_smooth)
        if do_get_frames:
            (_, self.frames_align, self.x_pos) = self.sda.frame_generation(self.signals.x_dly, self.signals.x_sda, self.signals.x_thr)
        else:
            self.signals.x_pos = self.sda.frame_position(self.signals.x_sda, self.signals.x_thr)

    def __do_sda(self, xin: np.ndarray, mode_sda: int, mode_thr: int, do_smooth=False) -> None:
        # --- Performing SDA
        if mode_sda == 0:
            xsda = self.sda.sda_norm(xin)
            text_sda = 'Normal'
        elif mode_sda == 1:
            xsda = self.sda.sda_neo(xin)
            text_sda = 'NEO'
        elif mode_sda == 2:
            xsda = self.sda.sda_mteo(xin)
            text_sda = 'MTEO'
        elif mode_sda == 3:
            xsda = self.sda.sda_aso(xin)
            text_sda = 'ASO'
        elif mode_sda == 4:
            xsda = self.sda.sda_ado(xin)
            text_sda = 'ADO'
        elif mode_sda == 5:
            xsda = self.sda.sda_eed(xin, self.signals.fs_adc)
            text_sda = 'EED'
        elif mode_sda == 6:
            xsda = self.sda.sda_spb(xin)
            text_sda = 'SPB'
        else:
            xsda = np.zeros(shape=xin.shape)
            text_sda = 'Err'

        self.signals.x_sda = self.sda.sda_smooth(xsda)[0] if do_smooth else xsda

        # --- Performing Thresholding
        if mode_thr == 0:
            xthr = self.sda.thres_const(xsda)
            text_thr = 'CONST'
        elif mode_thr == 1:
            xthr = self.sda.thres_mad(xsda)
            text_thr = 'MAD'
        elif mode_thr == 2:
            xthr = self.sda.thres_rms(xsda)
            text_thr = 'RMS'
        elif mode_thr == 3:
            xthr = self.sda.thres_blackrock(xsda)
            text_thr = 'RMS_BL'
        elif mode_thr == 4:
            xthr = self.sda.thres_ma(xsda)
            text_thr = 'MA'
        elif mode_thr == 5:
            xthr = self.sda.thres_winsorization(xsda)
            text_thr = 'Wins'
        elif mode_thr == 6:
            xthr = self.sda.thres_salvan_golay(xsda)
            text_thr = 'SG'
        else:
            xthr = np.zeros(shape=xsda.shape)
            text_thr = 'Err'

        self.signals.x_thr = xthr
        addon = "+Smooth" if do_smooth else ""
        self.used_methods = f'{text_sda}+{text_thr}{addon}'
