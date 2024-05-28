from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.dly_amp import DlyAmp, SettingsDLY
from package.analog.comp import Comp, SettingsCMP
from package.analog.dev_load import ElectricalLoad, SettingsDEV
from package.analog.cur_amp import CurrentAmplifier, SettingsCUR
from package.analog.int_ana import IntegratorStage, SettingsINT
from package.analog.adc_sar import SettingsADC, ADC_SAR
from src_mem.pipeline_mem import PipelineSignal


class _SettingsPipe:
    """Settings class for setting-up the pipeline"""
    __vdd = 5.0
    __vss = -5.0
    __en_noise = False
    __dly_bottom = 0.1e-3

    # --- Define Settings
    __settings_amp = SettingsAMP(
        vss=__vss, vdd=__vdd,
        fs_ana=0.0,
        gain=40,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6,
        noise_en=__en_noise,
        noise_edev=100e-9,
        f_chop=10e3
    )
    __settings_dly = SettingsDLY(
        vss=__vss, vdd=__vdd,
        fs_ana=0.0,
        t_dly=__dly_bottom,
        offset=0e-6,
        noise_en=__en_noise,
        noise_edev=100e-9
    )
    __settings_cmp = SettingsCMP(
        vss=__vss, vdd=__vdd,
        out_analog=False,
        gain=100,
        offset=0e-3,
        noise=__en_noise
    )
    __settings_load0 = SettingsDEV(
        type='R',
        fs_ana=0.0,
        noise_en=__en_noise,
        para_en=False,
        dev_value=10e3,
        temp=300
    )
    __settings_load1 = SettingsDEV(
        type='Dd',
        fs_ana=0.0,
        noise_en=__en_noise,
        para_en=False,
        dev_value=10e3,
        temp=300
    )
    __settings_cur = SettingsCUR(
        vss=__vss, vdd=__vdd,
        fs_ana=0.0,
        transimpedance=1e3,
        offset_v=1e-6, offset_i=1e-12,
        noise_en=__en_noise,
        para_en=False
    )
    __settings_int = SettingsINT(
        vss=__vss, vdd=__vdd,
        tau=100e-3,
        res_in=10e3,
        offset_v=1e-3,
        offset_i=1e-9,
        do_invert=False,
        noise_en=__en_noise,
        noise_edev=10e-9
    )
    __settings_adc = SettingsADC(
        vss=__vss, vdd=__vdd,
        type_out="signed",
        dvref=0.1,
        fs_ana=0.0,
        fs_dig=20e3, osr=1, Nadc=12
    )

    def __init__(self, fs_ana: float):
        self.__settings_amp.fs_ana = fs_ana
        self.__settings_dly.fs_ana = fs_ana
        self.__settings_adc.fs_ana = fs_ana
        self.__settings_cur.fs_ana = fs_ana
        self.__settings_load0.fs_ana = fs_ana
        self.__settings_load1.fs_ana = fs_ana

        # --- Init Pipeline Elements
        self._preamp = PreAmp(self.__settings_amp)
        self._dlyamp = DlyAmp(self.__settings_dly)
        self._cmp = Comp(self.__settings_cmp)
        self._load0 = ElectricalLoad(self.__settings_load0)
        self._load1 = ElectricalLoad(self.__settings_load1)
        self._curamp = CurrentAmplifier(self.__settings_cur)
        self._int = IntegratorStage(self.__settings_int, fs_ana)
        self._adc = ADC_SAR(self.__settings_adc)


class Pipeline(PipelineCMD, _SettingsPipe):
    """Processing Pipeline for analysing invasive neural activities"""
    signals: PipelineSignal

    def __init__(self, fs_ana: float):
        PipelineCMD.__init__(self)
        _SettingsPipe.__init__(self, fs_ana)

        self._path2pipe = abspath(__file__)
        self.generate_folder('runs', '_mem')

    def prepare_saving(self) -> dict:
        """Getting processing data of selected signals"""
        mdict = {"fs_ana": self.signals.fs_ana,
                 "fs_adc": self.signals.fs_adc,
                 "fs_dig": self.signals.fs_dig,
                 "u_inp": self.signals.u_inp,
                 "u_inn": self.signals.u_inn,
                 "x_feat": self.signals.x_feat}
        return mdict

    def do_plotting(self, data: PipelineSignal, channel: int) -> None:
        """Function to plot results"""
        pass

    def run(self, uinp: np.ndarray, u_offset: np.ndarray) -> None:
        x0 = PipelineSignal()
        x0.fs_ana = self._preamp.settings.fs_ana
        x0.fs_adc = self._adc.settings.fs_adc
        x0.fs_dig = self._adc.settings.fs_dig

        x0.u_inp = uinp
        x0.u_inn = np.array(self._preamp.vcm)
        # ---- Configuration of pipeline
        x0.u_pre = self._preamp.pre_amp(x0.u_inp, x0.u_inn)
        x0.u_dly = self._dlyamp.do_simple_delay(x0.u_pre)
        x0.u_cmp = self._cmp.cmp_normal(x0.u_pre, x0.u_inn)
        x0.u_mem_top = x0.u_pre + u_offset
        x0.u_mem_bot = x0.u_dly

        x0.i_off0 = self._load0.get_current_response(u_offset, self._dlyamp.vcm)
        x0.i_load0 = self._load0.get_current_response(x0.u_mem_top, x0.u_mem_bot)
        x0.i_off1 = self._load1.get_current_response(u_offset, self._dlyamp.vcm)
        x0.i_load1 = self._load1.get_current_response(x0.u_mem_top, x0.u_mem_bot)

        x0.u_trans0 = self._curamp.transimpedance_amplifier(x0.i_load0 - x0.i_off0, 0.0)
        x0.u_trans1 = self._curamp.transimpedance_amplifier(x0.i_load1 - x0.i_off1, 0.0)
        xfeat0 = self._adc.adc_ideal(x0.u_trans0[-1])[0]
        xfeat1 = self._adc.adc_ideal(x0.u_trans1[-1])[0]
        x0.x_feat = np.append(xfeat0, xfeat1)

        # --- Return
        self.signals = x0
