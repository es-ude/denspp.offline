import numpy as np
from unittest import TestCase, main
from denspp.offline.pipeline.pipeline_handler import start_pipeline_processing
from denspp.offline.data_call.call_handler import SettingsData
from denspp.offline.template.call_data import DataLoader
from denspp.offline.template.pipeline_v0 import Pipeline


test_settings_1d = SettingsData(
    path='data',
    data_set='test_1d',
    data_case=0, data_point=0,
    t_range=[], ch_sel=[],
    fs_resample=20e3,
    do_mapping=True,
    is_mapping_str=False
)

test_settings_2d = SettingsData(
    path='data',
    data_set='test_2d',
    data_case=0, data_point=0,
    t_range=[], ch_sel=[],
    fs_resample=20e3,
    do_mapping=True,
    is_mapping_str=False
)


# --- Info: Function have to start with test_*
class TestPipeProcess(TestCase):
    def test_pipeline_access_data_1d(self):
        dut = DataLoader(settings=test_settings_1d)
        dut.do_call()
        data = dut.get_data()
        np.testing.assert_equal(data.data_raw.shape, [1, 10000])

    def test_pipeline_process_direct_1d(self):
        dut = DataLoader(settings=test_settings_1d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = Pipeline(test_settings_1d.fs_resample).run(data.data_raw[0])
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])
        except:
            self.assertTrue(False)

    def test_pipeline_process_wrapper_1d(self):
        try:
            start_pipeline_processing(
                object_dataloader=DataLoader,
                object_pipeline=Pipeline,
                en_testmode=True,
                sets_load_data=test_settings_1d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_pipeline_access_data_2d(self):
        dut = DataLoader(settings=test_settings_2d)
        dut.do_call()
        data = dut.get_data()
        np.testing.assert_equal(data.data_raw.shape, [4, 10000])

    def test_pipeline_process_direct_2d(self):
        dut = DataLoader(settings=test_settings_2d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = Pipeline(test_settings_2d.fs_resample).run(data.data_raw[0])
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])
        except:
            self.assertTrue(False)

    def test_pipeline_process_wrapper_2d(self):
        try:
            start_pipeline_processing(
                object_dataloader=DataLoader,
                object_pipeline=Pipeline,
                en_testmode=True,
                sets_load_data=test_settings_2d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

if __name__ == '__main__':
    main()
