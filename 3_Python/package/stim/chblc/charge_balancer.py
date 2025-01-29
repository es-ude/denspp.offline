from dataclasses import dataclass
from package.data_generator.waveform_generator import WaveformGenerator
import matplotlib.pyplot as plt

from enum import Enum
from package.analog.amplifier.comp import *


class CBEval(Enum):
    WINDOW = 1
    PROPORTIONAL = 2
    SIGMOID = 3
    REAL_WINDOW = 4

class CBStrat(Enum):
    OFFSET_CURRENT = 1
    ANO_DCLIPPING = 2
    ANO_DURATION = 3
    ANO_AMPLITUDE = 4
    ANO_ACLIPPING = 5
    NONE = 0

@dataclass
class CBSettings: # --change this!!!--
    """"""
    cbstrat: CBStrat # parameter to be changed
    cbeval: CBEval # evaluation scheme
    memory: bool # (window comp only) does waveform revert to original when value is within window?
    size: float # size of window comparator/linear factor
    charge: bool # adjustment is based on parameter/charge?
    adjust: float # how much the parameter/charge is adjusted

@dataclass
class WFGSettings:
    i_amp: float
    f_samp: float
    f_samp_wfg: float
    t_start: float
    t_end: float
    t_sine: float
    n_waves: int
    q_step: float


class ChargeBalancer:
    def __init__(self, wfgsettings: WFGSettings, cbsettings: CBSettings):
        self.i_amp = wfgsettings.i_amp
        self.f_samp = wfgsettings.f_samp
        self.f_samp_wfg = wfgsettings.f_samp_wfg
        self.t_start = wfgsettings.t_start
        self.t_end = wfgsettings.t_end
        self.t_sine = wfgsettings.t_sine
        self.n_waves = wfgsettings.n_waves
        self.q_step = wfgsettings.q_step
        
        self.t_clip = float(0)
        self.i_offset = float(0)
        # self.amp_ratio = float(1)
        self.i_amp_cat = self.i_amp
        self.i_amp_ano = self.i_amp
        self.i_lim = self.i_amp

        self.t_cat = self.t_sine / 2
        self.t_ano = self.t_cat

        self.cbstrat = cbsettings.cbstrat
        # self.adaptive = cbsettings.adaptive
        # self.window = cbsettings.window
        self.cbeval = cbsettings.cbeval
        self.memory = cbsettings.memory
        self.size = cbsettings.size
        self.charge = cbsettings.charge
        self.adjust = cbsettings.adjust

        self.q_delta = float(0)

        self.q_ano_init = 2 * self.t_ano * self.i_amp_ano / np.pi

        self.wfg = WaveformGenerator(self.f_samp_wfg)
        self.waveform = self.__update_waveform()

        if self.cbeval == CBEval.REAL_WINDOW:
            self.comp = Comp(RecommendedSettingsCMP)

    def get_t_rep(self):
        return self.t_start + self.n_waves * self._get_t_wave() + self.t_end
    
    def get_q_ano_init(self):
        return self.q_ano_init

    def _get_t_wave(self):
        return self.t_cat + self.t_ano

    def __update_waveform(self):
        tp = []
        td = []
        ws = []
        pc = []

        for i in range(self.n_waves):
            tp.append(self.t_start + i * self._get_t_wave())
            td.append(self.t_cat)
            ws.append(3)
            pc.append(True)
            
            tp.append(self.t_start + i * self._get_t_wave() + self.t_cat)
            td.append(self.t_ano)
            ws.append(3)
            pc.append(False)

            if self.t_clip:
                tp.append(self.t_start + i * self._get_t_wave() + self.t_cat + self.t_ano - self.t_clip)
                td.append(self.t_clip)
                ws.append(13)
                pc.append(False)

        tp.append(self.t_start + self.n_waves * self._get_t_wave())
        td.append(self.t_end)
        ws.append(13)
        pc.append(False)

        t, waveform = self.wfg.generate_waveform(tp, td, ws, pc)

        # if self.amp_ratio != 1:
        #     # ano = [self.amp_ratio if i > 0 else 1 for i in waveform]
        #     # waveform *= ano
        #     waveform = np.where(waveform > 0, waveform * self.amp_ratio, waveform)

        # waveform *= self.i_amp

        if self.i_amp_cat != self.i_amp_ano:
            waveform = np.where(waveform > 0, waveform * self.i_amp_ano, waveform * self.i_amp_cat)
        else:
            waveform *= self.i_amp

        if self.i_lim < self.i_amp:
            waveform = np.where(waveform > self.i_lim, self.i_lim, waveform)

        # noise = np.random.normal(0, 0.01*self.i_amp, len(waveform))
        # noise = np.where(waveform == 0, 0, noise)

        waveform += self.i_offset

        waveform = np.where(t < self.t_start, 0, waveform)
        waveform = np.where(t >= (self.t_start + self.n_waves * self._get_t_wave()), 0, waveform)

        if self.q_step != 0:
            # truncate = lambda x: int(x / self.q_step) * self.q_step
            # waveform = np.array([truncate(x) for x in waveform])
            waveform = np.where(waveform > 0, 
                                np.floor(waveform / self.q_step) * self.q_step,
                                np.ceil(waveform / self.q_step) * self.q_step)

        return waveform
    
    def perform_charge_balancing(self, voltage: float):
        # match self.cbstrat:
        #     case CBStrat.OFFSET_CURRENT:
        #         self.__offset_current(voltage)
        #     case CBStrat.ANO_DCLIPPING:
        #         self.__ano_dclipping(voltage)
        #     case CBStrat.ANO_DURATION:
        #         self.__ano_duration(voltage)
        #     case CBStrat.ANO_AMPLITUDE:
        #         self.__ano_amplitude(voltage)
        #     case CBStrat.ANO_ACLIPPING:
        #         self.__ano_aclipping(voltage)
        #     case _:
        #         print("Invalid strategy!")

        if self.cbstrat == CBStrat.NONE:
            pass
        else:
            factor = self.__evaluate(voltage)
            change = factor * self.adjust

            if self.memory or self.cbeval == CBEval.WINDOW:
                self.q_delta += change
            else:
                self.q_delta = change

            if self.cbeval == CBEval.WINDOW and not self.memory and change == 0:
                self.q_delta = 0

            if self.charge:
                self.__adjust_waveform(self.__to_parameter(self.q_delta))
            else:
                self.__adjust_waveform(self.q_delta)

            self.waveform = self.__update_waveform()

    def __evaluate(self, voltage):
        match self.cbeval:
            case CBEval.WINDOW:
                if voltage > self.size:
                    return 1
                elif voltage < -self.size:
                    return -1
                else:
                    return 0
            case CBEval.PROPORTIONAL:
                return voltage * self.size
            case CBEval.REAL_WINDOW:
                over = self.comp.cmp_normal(voltage, self.size)
                under = self.comp.cmp_normal(-self.size, voltage)
                if over and under:
                    print("comp error!")
                    return 0
                elif over and not under:
                    return 1
                elif not over and under:
                    return -1
                else:
                    return 0
            case _:
                print("EVAL ERROR!")
                return 0
            
    def __to_parameter(self, q_delta):
        match self.cbstrat:
            case CBStrat.OFFSET_CURRENT:
                return q_delta / (self.n_waves * self._get_t_wave())
            case CBStrat.ANO_DCLIPPING:
                q_delta /= self.n_waves
                return self.t_ano * np.arccos(1 - np.pi * q_delta / (self.t_ano * self.i_amp_ano)) / np.pi
            case CBStrat.ANO_DURATION:
                q_delta /= self.n_waves
                return q_delta * np.pi / (2 * self.i_amp_ano)
            case CBStrat.ANO_AMPLITUDE:
                q_delta /= self.n_waves
                return q_delta * np.pi / (2 * self.t_ano)
            case CBStrat.ANO_ACLIPPING:
                if q_delta == 0:
                    return self.i_amp_ano
                else:
                    # print(q_delta)
                    q_delta /= self.n_waves
                    q = self.q_ano_init - q_delta
                    d1 = dx_1 = float(0)
                    d2 = dx = self.i_amp_ano
                    while dx != dx_1:
                        dx_1 = dx
                        dx = d1 - self.__get_q_error(d1, q) * ((d2 - d1) / (self.__get_q_error(d2, q) - self.__get_q_error(d1, q)))
                        ei = self.__get_q_error(dx, q)
                        if ei < 0:
                            d1 = dx
                        else:
                            d2 = dx
                        # print("dx =", dx, "e(dx) =", ei)
                    # print(self.__get_q_ano_aclipped(dx), self.q_ano_init)
                    return dx
            case _:
                print("STRAT ERROR")

    def __get_q_ano_aclipped(self, x):
        # print(x, self.t_ano)
        return self.t_ano * self.i_amp_ano / np.pi * (2 * (1 - np.sqrt(1 - (x / self.i_amp_ano) ** 2)) + x / self.i_amp_ano * (np.pi - 2 * np.arcsin(x / self.i_amp_ano)))
    
    def __get_q_error(self, x, q):
        return self.__get_q_ano_aclipped(x) - q
    
    def __adjust_waveform(self, param_change):
        match self.cbstrat:
            case CBStrat.OFFSET_CURRENT:
                self.i_offset = -param_change
                print("i_offset =", (self.i_offset * 1e6)@u_uA)
            case CBStrat.ANO_DCLIPPING:
                self.t_clip = param_change
                if self.t_clip > self.t_ano:
                    self.t_clip = self.t_ano
                if self.t_clip < 0:
                    self.t_clip = 0
                print("t_clip =", (self.t_clip * 1e6)@u_us)
            case CBStrat.ANO_DURATION:
                self.t_ano = self.t_sine / 2 - param_change
                if self.t_ano < 0:
                    self.t_ano = 0
                print("t_ano =", (self.t_ano * 1e6)@u_us)
            case CBStrat.ANO_AMPLITUDE:
                self.i_amp_ano = self.i_amp - param_change
                if self.i_amp_ano < 0:
                    self.i_amp_ano = 0
                print("i_amp_ano =", (self.i_amp_ano * 1e6)@u_uA)
            case CBStrat.ANO_ACLIPPING:
                self.i_lim = param_change
                if self.i_lim > self.i_amp_ano:
                    self.i_lim = self.i_amp_ano
                if self.i_lim < 0:
                    self.i_lim = 0
                print("i_lim =", (self.i_lim * 1e6)@u_uA)
            case _:
                print("STRAT ERROR")

    def __offset_current(self, voltage):
        if voltage > self.window:
            self.i_offset -= self.adjust
            print("i_offset decreased,", self.i_offset)
        elif voltage < -self.window:
            self.i_offset += self.adjust
            print("i_offset increased,", self.i_offset)

    def __ano_dclipping(self, voltage):
        if voltage > self.window and self.t_clip < self.t_ano:
            self.t_clip += self.adjust
            print("t_clip increased,", self.t_clip)
        elif voltage < -self.window and self.t_clip > 0:
            self.t_clip -= self.adjust
            print("t_clip decreased,", self.t_clip)

    def __ano_duration(self, voltage):
        if voltage > self.window and self.t_ano > 0:
            self.t_ano -= self.adjust
            print("t_ano decreased,", self.t_ano)
        elif voltage < -self.window:
            self.t_ano += self.adjust
            print("t_ano increased,", self.t_ano)

    def __ano_amplitude(self, voltage):
        if voltage > self.window and self.i_amp_ano > 0:
            self.i_amp_ano -= self.adjust * self.i_amp
            print("i_amp_ano decreased,", self.i_amp_ano)
        elif voltage < -self.window:
            self.i_amp_ano += self.adjust * self.i_amp
            print("i_amp_ano increased,", self.i_amp_ano)
        if self.i_amp_ano <= 0:
            self.i_amp_ano = 0

    def __ano_aclipping(self, voltage):
        if voltage > self.window and self.i_lim > 0:
            self.i_lim -= self.adjust
            print("i_lim decreased,", self.i_lim)
        elif voltage < -self.window and self.i_lim < self.i_amp:
            self.i_lim += self.adjust
            print("i_lim increased,", self.i_lim)
        if self.i_lim == 0:
            self.i_lim = self.adjust 
        if self.i_lim > self.i_amp:
            self.i_lim = self.i_amp       

if __name__ == "__main__":
    cbsettings = CBSettings(
        cbstrat=CBStrat.OFFSET_CURRENT,
        cbeval=CBEval.WINDOW,
        memory=True,
        size=0.04,
        charge=False,
        adjust=0.1*12e-6
    )
    wfgsettings = WFGSettings(
        i_amp=12e-6,
        f_samp=10e6,
        f_samp_wfg=200*10e6,
        t_start=1e-3,
        t_end=1e-3,
        t_sine=2e-3,
        n_waves=2
    )
    cbal = ChargeBalancer(wfgsettings, cbsettings)

    print(cbal.get_t_rep())

    plt.plot(cbal.waveform)

    cbal.perform_charge_balancing(0.05)

    plt.plot(cbal.waveform)
    ax = plt.gca()
    ax.grid()
    plt.tight_layout()
    plt.show()
