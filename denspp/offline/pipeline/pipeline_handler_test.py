import numpy as np
from unittest import TestCase, main
from denspp.offline.template.call_data_dummy import DataLoaderTest
from denspp.offline.pipeline.pipeline_handler import select_process_pipeline, start_processing_pipeline
from denspp.offline.data_call.call_handler import SettingsData
from denspp.offline.template.pipeline_v0 import PipelineV0


test_settings_1d = SettingsData(
    pipeline='PipelineV0',
    do_merge=False,
    path='data',
    data_set='test_1d',
    data_case=0, data_point=0,
    t_range_sec=[], ch_sel=[],
    fs_resample=20e3,
    do_mapping=True,
    is_mapping_str=False
)

test_settings_1d_merge = SettingsData(
    pipeline='PipelineV0',
    do_merge=True,
    path='data',
    data_set='test_1d',
    data_case=0, data_point=0,
    t_range_sec=[], ch_sel=[],
    fs_resample=20e3,
    do_mapping=True,
    is_mapping_str=False
)

test_settings_2d = SettingsData(
    pipeline='PipelineV0',
    do_merge=False,
    path='data',
    data_set='test_2d',
    data_case=0, data_point=0,
    t_range_sec=[], ch_sel=[],
    fs_resample=20e3,
    do_mapping=True,
    is_mapping_str=False
)


# --- Info: Function have to start with test_*
class TestPipeProcess(TestCase):
    def test_pipeline_access_data_1d(self):
        dut = DataLoaderTest(settings=test_settings_1d)
        dut.do_call()
        data = dut.get_data()
        np.testing.assert_equal(data.data_raw.shape, [1, int(data.fs_orig)])

    def test_pipeline_process_direct_1d(self):
        dut = DataLoaderTest(settings=test_settings_1d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = PipelineV0(test_settings_1d.fs_resample).run(data.data_raw[0])
        except:
            self.assertTrue(False)
        else:
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])

    def test_pipeline_process_wrapper_1d(self):
        try:
            select_process_pipeline(
                object_dataloader=DataLoaderTest,
                object_pipeline=PipelineV0,
                sets_load_data=test_settings_1d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_pipeline_process_data_1d(self):
        dut = DataLoaderTest(settings=test_settings_1d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = PipelineV0(test_settings_2d.fs_resample).run(data.data_raw[0])
        except:
            self.assertTrue(False)
        else:
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])

    def test_pipeline_process_data_wrapper_1d(self):
        try:
            select_process_pipeline(
                object_dataloader=DataLoaderTest,
                object_pipeline=PipelineV0,
                sets_load_data=test_settings_1d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_pipeline_access_data_2d(self):
        dut = DataLoaderTest(settings=test_settings_2d)
        dut.do_call()
        data = dut.get_data()
        np.testing.assert_equal(data.data_raw.shape, [4, int(data.fs_orig)])

    def test_pipeline_process_direct_2d(self):
        dut = DataLoaderTest(settings=test_settings_2d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = PipelineV0(test_settings_2d.fs_resample).run(data.data_raw[0])
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])
        except:
            self.assertTrue(False)

    def test_pipeline_process_wrapper_2d(self):
        try:
            select_process_pipeline(
                object_dataloader=DataLoaderTest,
                object_pipeline=PipelineV0,
                sets_load_data=test_settings_2d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_pipeline_merge_data_2d(self):
        dut = DataLoaderTest(settings=test_settings_2d)
        dut.do_call()
        data = dut.get_data()
        try:
            rslt = PipelineV0(test_settings_2d.fs_resample).run(data.data_raw[0])
        except:
            self.assertTrue(False)
        else:
            np.testing.assert_equal(rslt['u_in'], data.data_raw[0])

    def test_pipeline_merge_data_wrapper_2d(self):
        try:
            select_process_pipeline(
                object_dataloader=DataLoaderTest,
                object_pipeline=PipelineV0,
                sets_load_data=test_settings_2d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(True)

    def test_entry_point(self):
        try:
            rslt = start_processing_pipeline(
                sets_load_data=test_settings_2d
            )
        except:
            self.assertTrue(False)
        else:
            self.assertTrue(len(rslt) == 3)


if __name__ == '__main__':
    main()
