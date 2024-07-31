from dataclasses import dataclass
from waveform_generator import WaveformGenerator
import matplotlib.pyplot as plt

import numpy as np

from enum import Enum

class CBStrat(Enum):
    OFFSET_CURRENT = 1
    ANO_DCLIPPING = 2
    ANO_DURATION = 3
    ANO_AMPLITUDE = 4
    ANO_ACLIPPING = 5

@dataclass
class CBSettings:
    """"""
    cbstrat: CBStrat
    adaptive: bool
    window: float
    adjust: float

@dataclass
class WFGSettings:
    i_amp: float
    f_samp: float
    f_samp_wfg: float
    t_start: float
    t_end: float
    t_sine: float
    n_waves: int


class ChargeBalancer:
    def __init__(self, wfgsettings: WFGSettings, cbsettings: CBSettings):
        self.i_amp = wfgsettings.i_amp
        self.f_samp = wfgsettings.f_samp
        self.f_samp_wfg = wfgsettings.f_samp_wfg
        self.t_start = wfgsettings.t_start
        self.t_end = wfgsettings.t_end
        self.t_sine = wfgsettings.t_sine
        self.n_waves = wfgsettings.n_waves
        
        self.t_clip = float(0)
        self.i_offset = float(0)
        self.amp_ratio = float(1)
        self.i_lim = self.i_amp

        self.t_cat = self.t_sine / 2
        self.t_ano = self.t_cat

        self.cbstrat = cbsettings.cbstrat
        self.adaptive = cbsettings.adaptive
        self.window = cbsettings.window
        self.adjust = cbsettings.adjust

        self.wfg = WaveformGenerator(self.f_samp_wfg)
        self.waveform = self._generate_waveform()



    def get_t_rep(self):
        return self.t_start + self.n_waves * self._get_t_wave() + self.t_end

    def _get_t_wave(self):
        return self.t_cat + self.t_ano

    def _generate_waveform(self):
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

        if self.amp_ratio != 1:
            ano = [self.amp_ratio if i > 0 else 1 for i in waveform]
            waveform *= ano

        waveform *= self.i_amp

        if self.i_lim < self.i_amp:
            waveform = np.array([self.i_lim if i > self.i_lim else i for i in waveform])

        return self.i_offset + waveform
    
    def perform_charge_balancing(self, voltage: float):
        match self.cbstrat:
            case CBStrat.OFFSET_CURRENT:
                self.__offset_current(voltage)
            case CBStrat.ANO_DCLIPPING:
                self.__ano_dclipping(voltage)
            case CBStrat.ANO_DURATION:
                self.__ano_duration(voltage)
            case CBStrat.ANO_AMPLITUDE:
                self.__ano_amplitude(voltage)
            case CBStrat.ANO_ACLIPPING:
                self.__ano_aclipping(voltage)
            case _:
                print("Invalid strategy!")
        self.waveform = self._generate_waveform()

    def __offset_current(self, voltage):
        if voltage > self.window:
            print("exceeded 0.1V!")
            self.i_offset -= self.adjust
        elif voltage < -self.window:
            print("exceeded -0.1V!")
            self.i_offset += self.adjust

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
        if voltage > self.window and self.amp_ratio > 0:
            self.amp_ratio -= self.adjust
            print("amp_ratio decreased,", self.amp_ratio)
        elif voltage < -self.window:
            self.amp_ratio += self.adjust
            print("amp_ratio increased,", self.amp_ratio)
        if self.amp_ratio <= 0:
            self.amp_ratio = self.adjust

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
        adaptive=False,
        window=0.04,
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


