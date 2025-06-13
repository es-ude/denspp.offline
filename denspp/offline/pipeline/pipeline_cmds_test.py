import numpy as np
from os.path import join, exists
from unittest import TestCase, main
from denspp.offline.pipeline.pipeline_cmds import PipelineCMD, PipelineLibrary, DataloaderLibrary
from denspp.offline.pipeline.pipeline_handler_test import test_settings_1d
from denspp.offline.pipeline.call_data_test import DataLoaderTest, SettingsData


# --- Info: Function have to start with test_*
class TestPipeCommands(TestCase):
    dut = PipelineCMD()

    def test_pipecmd_get_name(self):
        name = self.dut.get_pipeline_name()
        self.assertEqual(name, 'PipelineCMD')

    def test_pipecmd_generate_folder(self):
        self.dut.generate_run_folder(
            path2runs=join('temp_test', 'runs'),
            addon='_test'
        )
        self.assertTrue(exists(self.dut.path2save))

    def test_pipecmd_save_results(self):
        self.dut.generate_run_folder(
            path2runs=join('temp_test', 'runs'),
            addon='_test'
        )
        self.dut.save_results(
            name='test',
            data=dict({'name': 'A', 'value': 1})
        )
        self.assertTrue(exists(self.dut.path2save))

    def test_pipecmd_apply_mapping_shape_wo_zero(self):
        mapping = np.array([[1, 2], [3, 4]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        chck = data_mapd.shape == (2, 2, int(test_data.fs_orig)) and np.sum(data_mapd == 0.) == 0
        self.assertTrue(chck)

    def test_pipecmd_apply_mapping_content_wo_zero(self):
        mapping = np.array([[1, 3], [2, 4]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        np.testing.assert_equal(data_mapd[1, 0, :], test_data.data_raw[1, :])

    def test_pipecmd_apply_mapping_shape_with_zero(self):
        mapping = np.array([[1, 2], [0, 3]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d_zero',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=20e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        chck = data_mapd.shape == (2, 2, int(test_data.fs_orig)) and np.sum(data_mapd == 0.) == test_data.data_raw.shape[-1]
        self.assertTrue(chck)

    def test_pipecmd_apply_mapping_content_with_zero(self):
        mapping = np.array([[1, 3], [0, 2]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d_zero',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=20e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        np.testing.assert_equal(data_mapd[1, 1, :], test_data.data_raw[1, :])

    def test_pipecmd_return_mapping_shape_wo_zero(self):
        mapping = np.array([[1, 2], [3, 4]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=40e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        data_ret = self.dut.deploy_mapping(
            data=data_mapd,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        chck = data_ret.shape == (4, int(test_data.fs_orig)) and np.sum(data_mapd == 0.) == 0
        self.assertTrue(chck)

    def test_pipecmd_return_mapping_shape_with_zero(self):
        mapping = np.array([[1, 2], [3, 0]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d_zero',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=40e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        data_ret = self.dut.deploy_mapping(
            data=data_mapd,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        chck = data_ret.shape == (3, int(test_data.fs_orig)) and np.sum(data_mapd == 0.) == int(test_data.fs_orig)
        self.assertTrue(chck)

    def test_pipecmd_return_mapping_content_wo_zero(self):
        mapping = np.array([[1, 3], [2, 4]])
        test_settings = SettingsData(
            pipeline='PipelineV0',
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range_sec=[0], ch_sel=[],
            fs_resample=40e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoaderTest(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        data_ret = self.dut.deploy_mapping(
            data=data_mapd,
            electrode_id=test_data.electrode_id,
            mapping=mapping
        )
        np.testing.assert_equal(data_ret, test_data.data_raw)

    def test_pipeline_library_available(self):
        pipelib = PipelineLibrary().get_registry()
        matches = [item for item in pipelib.get_library_overview() if 'Pipeline' in item]
        self.assertTrue(len(matches))

    def test_pipeline_library_access(self):
        pipelib = PipelineLibrary().get_registry()
        matches = [item for item in pipelib.get_library_overview() if 'Pipeline' in item]
        if not len(matches):
            self.assertTrue(False)
        else:
            datalib = DataloaderLibrary().get_registry()
            data_handler = datalib.build('DataLoaderTest', settings=test_settings_1d)
            data_handler.do_call()
            data = data_handler.get_data()

            pipe = pipelib.build('PipelineV0', fs_ana=data.fs_orig)
            rslt = pipe.run(data.data_raw[0, :])
            self.assertTrue(rslt['u_in'].shape == (int(data.fs_orig), ))

    def test_dataloader_library_available(self):
        datalib = DataloaderLibrary().get_registry()
        matches = [item for item in datalib.get_library_overview() if 'DataLoader' in item]
        self.assertTrue(len(matches))

    def test_dataloader_library_access(self):
        datalib = DataloaderLibrary().get_registry()
        ovr = datalib.get_library_overview()
        if not len(ovr):
            self.assertTrue(False)
        else:
            data_handler = datalib.build('DataLoaderTest', settings=test_settings_1d)
            data_handler.do_call()
            data = data_handler.get_data()
            self.assertTrue(data.data_raw.shape == (1, int(data.fs_orig)))


if __name__ == '__main__':
    main()
