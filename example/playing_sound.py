import numpy as np
from denspp.player.devices.sound_card import TranslatorSoundCard


if __name__ == "__main__":
    dut = TranslatorSoundCard()
    do_play = True
    window = 10.

    if not do_play:
        a = dut.get_data(duration_sec=window)
        dut.plot_data(a)
    else:
        time = np.linspace(start=0, stop=window, num=int(window * dut.get_sampling_rate))
        signal = np.sin(2*np.pi*time*200.) + np.random.randn(*time.shape) * 0.01
        dut.play_data(signal)
