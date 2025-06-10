import numpy as np
from os.path import join, exists
from unittest import TestCase, main
from denspp.offline.pipeline.pipeline_cmds import PipelineCMD
from denspp.offline.template.call_data import DataLoader, SettingsData


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

    def test_pipecmd_apply_mapping_shape(self):
        test_settings = SettingsData(
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoader(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 2], [3, 4]])
        )
        chck = data_mapd.shape == (2, 2, 10000) and np.sum(data_mapd == 0.) == 0
        self.assertTrue(chck)

    def test_pipecmd_apply_mapping_content(self):
        test_settings = SettingsData(
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoader(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 3], [2, 4]])
        )
        np.testing.assert_equal(data_mapd[1, 0, :], test_data.data_raw[1, :])

    def test_pipecmd_return_mapping_shape(self):
        test_settings = SettingsData(
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoader(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 2], [3, 4]])
        )
        data_ret = self.dut.deploy_mapping(
            data=data_mapd,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 3], [2, 4]])
        )
        chck = data_ret.shape == (4, 10000) and np.sum(data_mapd == 0.) == 0
        self.assertTrue(chck)

    def test_pipecmd_return_mapping_content(self):
        test_settings = SettingsData(
            path='data',
            data_set='test_2d',
            data_case=0, data_point=0,
            t_range=[0], ch_sel=[],
            fs_resample=100e3,
            do_mapping=True,
            is_mapping_str=False
        )
        dut = DataLoader(settings=test_settings)
        dut.do_call()
        test_data = dut.get_data()
        data_mapd = self.dut.apply_mapping(
            data=test_data.data_raw,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 3], [2, 4]])
        )
        data_ret = self.dut.deploy_mapping(
            data=data_mapd,
            electrode_id=test_data.electrode_id,
            mapping=np.array([[1, 3], [2, 4]])
        )
        np.testing.assert_equal(data_ret, test_data.data_raw)


if __name__ == '__main__':
    main()
