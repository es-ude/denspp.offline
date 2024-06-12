from os.path import abspath
import numpy as np

from package.pipeline_cmds import PipelineCMD
from package.analog.pre_amp import PreAmp, SettingsAMP
from package.analog.dly_amp import DlyAmp, SettingsDLY
from package.analog.comp import Comp, SettingsCMP
# from package.analog.dev_load import ElectricalLoad, SettingsDEV
from package.analog.cur_amp import CurrentAmplifier, SettingsCUR
from package.analog.int_ana import IntegratorStage, SettingsINT
from package.analog.adc_sar import SettingsADC, ADC_SAR
from src_mem.memristor_extension import SettingsMem, MemristorModel
from src_mem.pipeline_mem import PipelineSignal

from src_mem.memristor_plots import plt_pipeline_signals_part_one, plt_pipeline_signals_part_two


def append_numpy_signals(data_in: np.ndarray, data_new: np.ndarray) -> np.ndarray:
    """Appending the numpy arrays easily
    Args:
        data_in:    Test
        data_new:   Test
    Returns:
        Numpy array with concatenated signals (num of channels in first dimension)
    """
    data_pre = data_new.reshape((1, data_new.size))
    if data_in is None:
        return data_pre
    else:
        return np.append(data_in, data_pre, axis=0)


class _SettingsPipe:
    __vdd = 5.0
    __vss = -5.0
    __en_noise = False

    # --- Define Settings
    __settings_amp = SettingsAMP(
        vss=__vss, vdd=__vdd,
        fs_ana=0.0,
        gain=1,
        n_filt=1, f_filt=[0.1, 8e3], f_type="band",
        offset=1e-6,
        noise_en=__en_noise,
        noise_edev=100e-9,
        f_chop=10e3
    )
    __settings_dly = SettingsDLY(
        vss=__vss, vdd=__vdd,
        fs_ana=0.0,
        t_dly=0.1e-3,
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
    __settings_load = SettingsMem(
        type='M1',
        fs_ana=0.0,
        noise_en=__en_noise,
        para_en=False,
        dev_value={},
        dev_branch=1,
        dev_sel=1,
        temp=300,
        area=0.045
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

    def __init__(self, fs_ana: float) -> None:
        """Settings class for setting-up the pipeline
        Args:
            fs_ana:     Sampling rate [Hz]
        Returns:
            None
        """
        self.__settings_amp.fs_ana = fs_ana
        self.__settings_dly.fs_ana = fs_ana
        self.__settings_adc.fs_ana = fs_ana
        self.__settings_cur.fs_ana = fs_ana
        self.__settings_load.fs_ana = fs_ana

        # --- Init Pipeline Elements (Init. of _load<X> later)
        self._preamp = PreAmp(self.__settings_amp)
        self._dlyamp = DlyAmp(self.__settings_dly)
        self._load = MemristorModel(self.__settings_load)
        self._cmp = Comp(self.__settings_cmp)
        self._curamp = CurrentAmplifier(self.__settings_cur)
        self._intamp = IntegratorStage(self.__settings_int, fs_ana)
        self._adc = ADC_SAR(self.__settings_adc)


class Pipeline(PipelineCMD, _SettingsPipe):
    signals: PipelineSignal

    def __init__(self, fs_ana: float) -> None:
        """Processing Pipeline for analysing invasive neural activities
        Args:
            fs_ana:         Sampling rate [Hz]
        Returns:
            None
        """
        PipelineCMD.__init__(self)
        _SettingsPipe.__init__(self, fs_ana)

        self._tpos_adc = [-1]
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

    def do_plotting(self, num_sample: int) -> None:
        """Function to plot results"""

        plt_pipeline_signals_part_one(self.signals, num_sample, path2save=self.path2save)
        plt_pipeline_signals_part_two(self.signals, num_sample, path2save=self.path2save)

    def define_time_adc_samp(self, tq: np.ndarray | list) -> None:
        """Definition of sampling timepoints for ADC quantization"""
        self._tpos_adc = tq

    def run(self, uinp: np.ndarray, u_offset: np.ndarray | list, t_dly: np.ndarray | list,
            gain=(), do_sum=False) -> None:
        """Running the pipeline
        Args:
            uinp:       Transient signal of each channel
            u_offset:   List or numpy array with offset voltage values for each analogue feature extraction stage
            t_dly:      List or numpy array with time delay values for each analogue feature extraction stage
            gain:       List or numpy array with gain value for each analogue feature extraction stage [+/- 0.8 ... 1.2]
            do_sum:     Do full integration of current
        Returns:
            None
        """
        self._dlyamp.settings.t_dly = t_dly

        x0 = PipelineSignal()
        x0.fs_ana = self._preamp.settings.fs_ana
        x0.fs_adc = self._adc.settings.fs_adc
        x0.fs_dig = self._adc.settings.fs_dig

        # --- Definition of input
        x0.u_inp = uinp
        x0.u_inn = np.array(self._preamp.vcm)
        # x0.u_pre = self._preamp.pre_amp(x0.u_inp, x0.u_inn)
        x0.u_pre = uinp
        x0.u_cmp = self._cmp.cmp_normal(x0.u_pre, x0.u_inn)

        # --- Definition of Load Data Processing
        num_feats = len(t_dly) if isinstance(t_dly, list) else t_dly.size
        x0.x_feat = np.zeros((num_feats, ), dtype=float)

        for idx in range(0, num_feats):
            self._dlyamp.settings.t_dly = t_dly[idx]
            gain0 = 1.0 if len(gain) == 0 else gain[idx]

            u_dly = self._dlyamp.do_simple_delay(x0.u_pre)
            u_mem_top = x0.u_pre
            u_mem_bot = gain0 * u_dly - u_offset[idx]

            i_load = self._load.get_current(u_mem_top, u_mem_bot)
            i_off = self._load.get_current(u_offset[idx], self._dlyamp.vcm)

            u_tra = self._curamp.push_pull_abs_amplifier(i_load - i_off)
            if do_sum:
                u_int = np.zeros((u_tra.size, len(self._tpos_adc)), dtype=float)
                for i, tq in enumerate(self._tpos_adc):
                    xpos = int(x0.fs_ana * tq)-1
                    u_int[:xpos, i] = self._intamp.do_ideal_integration(u_tra[:xpos], self._intamp.vcm)
                    u_int[xpos:, i] += u_int[xpos-1, i]
                x0.x_feat[idx] = u_int[-1, -1] / u_int[-1, 0] if len(self._tpos_adc) > 1 else u_int[xpos, -1]
                # x0.x_feat[idx] = self._adc.adc_ideal(u_int[-1])[0]
            else:
                u_int = np.zeros((len(self._tpos_adc, )))
                for i, tq in enumerate(self._tpos_adc):
                    xpos = int(x0.fs_ana * tq)
                    u_int[i] = self._intamp.do_ideal_integration_sample(u_tra[:xpos], self._intamp.vcm)
                x0.x_feat[idx] = u_int[-1] / u_int[0] if len(self._tpos_adc) > 1 else u_int[-1]
                # x0.x_feat[idx] = self._adc.adc_ideal(u_int[-1])[0]

            # --- Return signals to handler
            x0.u_dly = append_numpy_signals(x0.u_dly, u_dly)
            x0.u_mem_top = append_numpy_signals(x0.u_mem_top, u_mem_top)
            x0.u_mem_bot = append_numpy_signals(x0.u_mem_bot, u_mem_bot)
            x0.i_tra = append_numpy_signals(x0.i_tra, i_load - i_off)
            x0.u_tra = append_numpy_signals(x0.u_tra, u_tra)
            x0.u_int = append_numpy_signals(x0.u_int, u_int)

        # --- Return
        self.signals = x0
