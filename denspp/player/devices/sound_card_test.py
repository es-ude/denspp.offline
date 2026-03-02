import numpy as np
import unittest
from .sound_card import TranslatorSoundCard


class TestTranslatorSoundcard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dut = TranslatorSoundCard()

    def test_access(self):
        self.assertEqual(self.dut.get_sampling_rate, 44100)
        self.assertEqual(self.dut.get_voltage_range, (-1.5, 1.5))
        self.assertEqual(self.dut.get_bitwidth, 16)
        self.assertEqual(self.dut.get_voltage_resolution, 4.57763671875e-05)

    def test_available_microphone(self):
        rslt = self.dut.get_list_available_microphones
        self.assertGreaterEqual(len(rslt), 1)

    def test_available_speaker(self):
        rslt = self.dut.get_list_available_speaker
        self.assertGreaterEqual(len(rslt), 1)

    def test_update_rate(self):
        dut = TranslatorSoundCard()
        self.assertEqual(dut.get_sampling_rate, 44100)
        dut.update_sampling_rate(1000)
        self.assertEqual(dut.get_sampling_rate, 1000)

    def test_playing_data(self):
        fs = self.dut.get_sampling_rate
        time = np.linspace(start=0., stop=0.5, num=int(0.5* fs), endpoint=False)
        time.repeat(1, axis=-1)

        data = 0.5 * np.sin(2* np.pi* 20* time)
        self.dut.play_data(data)

    def test_getting_data(self):
        duration = 0.5
        num_samples = int(duration * self.dut.get_sampling_rate)+1
        data = self.dut.get_data(duration)
        self.assertEqual(data.shape, (num_samples, ))


if __name__ == '__main__':
    unittest.main()
