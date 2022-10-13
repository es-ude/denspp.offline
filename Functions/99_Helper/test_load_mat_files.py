from unittest import TestCase
import sys, os
import numpy as np

from load_mat_files import *

class Test(TestCase):
    def test_load_mat_file(self):
        folder = os.sep.join(os.path.dirname(os.path.abspath(__file__)).split(os.sep)[:-2] + ['2_Data', '01_SimDaten_Martinez2009'])
        file = 'simulation_1.mat'
        file_path = os.path.join(folder, file)
        data = load_mat_file(file_path)

    def test_load_01_SimDaten_Martinez2009(self):
        data_list = load_01_SimDaten_Martinez2009([1, 2])
        for data in data_list:
            self.assertEqual(list(data.keys()), ['sampling_rate', 'raw_data', 'spike_times', 'spike_cluster'])
            self.assertEqual(np.ndarray, type(data['raw_data']))
            self.assertEqual(np.ndarray, type(data['spike_times']))
            self.assertEqual(np.ndarray, type(data['spike_cluster']))
            self.assertEqual(float, type(data['sampling_rate']))
            self.assertEqual(np.float64, type(data['raw_data'][0]))
            self.assertEqual(np.int32, type(data['spike_times'][0]))
            self.assertEqual(np.uint8, type(data['spike_cluster'][0]))

    def test_load_02_SimDaten_Pedreira2012(self):
        data_list = load_02_SimDaten_Pedreira2012([1, 90])
        for data in data_list:
            self.assertEqual(list(data.keys()), ['sampling_rate', 'raw_data', 'spike_times', 'spike_cluster'])
            self.assertEqual(np.ndarray, type(data['raw_data']))
            self.assertEqual(np.ndarray, type(data['spike_times']))
            self.assertEqual(np.ndarray, type(data['spike_cluster']))
            self.assertEqual(float, type(data['sampling_rate']))
            self.assertEqual(np.float64, type(data['raw_data'][0]))
            self.assertEqual(np.int32, type(data['spike_times'][0]))
            self.assertEqual(np.uint8, type(data['spike_cluster'][0]))

    def test_load_03_SimDaten_Quiroga2020(self):
        data_list = load_03_SimDaten_Quiroga2020([16])
        for data in data_list:
            self.assertEqual(list(data.keys()), ['sampling_rate', 'raw_data', 'spike_times', 'spike_cluster'])
            self.assertEqual(np.ndarray, type(data['raw_data']))
            self.assertEqual(np.ndarray, type(data['spike_times']))
            self.assertEqual(np.ndarray, type(data['spike_cluster']))
            self.assertEqual(float, type(data['sampling_rate']))
            self.assertEqual(np.float64, type(data['raw_data'][0]))
            self.assertEqual(np.int32, type(data['spike_times'][0]))
            self.assertEqual(np.uint8, type(data['spike_cluster'][0]))

    def test_load_05_Daten_Klaes(self):
        load_05_Data_Klaes()