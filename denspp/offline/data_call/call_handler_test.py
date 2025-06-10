import numpy as np
from copy import deepcopy
from unittest import TestCase, main
from denspp.offline.data_call.call_handler import SettingsData
from denspp.offline.template.call_data import DataLoader


test_settings = SettingsData(
    path='data',
    data_set='test_1d',
    data_case=0, data_point=0,
    t_range=[0], ch_sel=[],
    fs_resample=100e3,
    do_mapping=True,
    is_mapping_str=False
)


# --- Info: Function have to start with test_*
class TestCallHandler(TestCase):
    def test_dataloader_1d_get_data(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_1d'

        dut = DataLoader(settings=settings)
        dut.do_call()
        data = dut.get_data()
        chck = data.data_type == 'Test_1d' and data.data_raw.shape == (1, 10000)
        self.assertTrue(chck)

    def test_dataloader_1d_meta(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_1d'

        dut = DataLoader(settings=settings)
        dut.do_call()
        try:
            dut.output_meta()
        except:
            self.assertFalse(False)
        else:
            self.assertTrue(True)

    def test_dataloader_1d_upsampling(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_1d'
        settings.fs_resample = 20e3

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_resample()
        data = dut.get_data()
        chck = data.data_raw.shape == (1, 20000) and data.time_end == 1.0
        self.assertTrue(chck)

    def test_dataloader_1d_cutting(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_1d'
        settings.t_range = [0.5, 0.8]

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_cut()
        data = dut.get_data()
        chck = data.data_raw.shape == (1, 3000) and data.time_end == 0.3
        self.assertTrue(chck)

    def test_dataloader_1d_cutting_resampling(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_1d'
        settings.fs_resample = 20e3
        settings.t_range = [0.5, 0.8]

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_resample()
        dut.do_cut()
        data = dut.get_data()
        chck = data.data_raw.shape == (1, 6000) and data.time_end == 0.3
        self.assertTrue(chck)

    def test_dataloader_2d_get_data(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_2d'

        dut = DataLoader(settings=settings)
        dut.do_call()
        data = dut.get_data()
        chck = data.data_type == 'Test_2d' and data.data_raw.shape == (4, 10000)
        self.assertTrue(chck)

    def test_dataloader_2d_meta(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_2d'

        dut = DataLoader(settings=settings)
        dut.do_call()
        try:
            dut.output_meta()
        except:
            self.assertFalse(False)
        else:
            self.assertTrue(True)

    def test_dataloader_2d_upsampling(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_2d'
        settings.fs_resample = 20e3

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_resample()
        data = dut.get_data()
        chck = data.data_raw.shape == (4, 20000) and data.time_end == 1.0
        self.assertTrue(chck)

    def test_dataloader_2d_cutting(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_2d'
        settings.t_range = [0.5, 0.8]

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_cut()
        data = dut.get_data()
        chck = data.data_raw.shape == (4, 3000) and data.time_end == 0.3
        self.assertTrue(chck)

    def test_dataloader_2d_cutting_resampling(self):
        settings = deepcopy(test_settings)
        settings.data_set = 'test_2d'
        settings.fs_resample = 20e3
        settings.t_range = [0.5, 0.8]

        dut = DataLoader(settings=settings)
        dut.do_call()
        dut.do_resample()
        dut.do_cut()
        data = dut.get_data()
        chck = data.data_raw.shape == (4, 6000) and data.time_end == 0.3
        self.assertTrue(chck)


if __name__ == '__main__':
    main()
