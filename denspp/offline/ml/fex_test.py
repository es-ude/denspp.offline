import os
import tempfile
import unittest
from copy import deepcopy

import joblib
import numpy as np

from .fex import DefaultSettingsFeature, FeatureExtraction, SettingsFeature


class FeatExtractionTest(unittest.TestCase):
    def setUp(self):

        self.set0: SettingsFeature = deepcopy(DefaultSettingsFeature)
        self.set0.num_features = 3
        self.dut = FeatureExtraction(settings=self.set0)

        # Create dummy data (20 Samples, 50 data points per frame)
        # 20 min as UMAP usually uses n_neighbors = 15
        np.random.seed(42)
        self.dummy_frames = np.random.rand(20, 50)

    def test_pdac_min_max(self):
        """Tests PDAC functions using one simpe frame."""
        simple_frame = np.array([[2, 4, 1, 3, 5]])

        # --- Test PDAC MIN ---

        expected_min = np.array([[4, 6, 5, 1]])
        out_min = self.dut.pdac_min(simple_frame)
        np.testing.assert_array_equal(out_min, expected_min)

        # --- Test PDAC MAX ---

        expected_max = np.array([[10, 0, 5, 1]])
        out_max = self.dut.pdac_max(simple_frame)
        np.testing.assert_array_equal(out_max, expected_max)

    def test_pca_shapes(self):
        """Tests dimensions of PCA results"""
        out_full = self.dut.pca_full(self.dummy_frames)
        self.assertEqual(out_full.shape, (20, 3))

        out_custom = self.dut.pca_custom(self.dummy_frames, svd_solver_mode="randomized", random_state=42)
        self.assertEqual(out_custom.shape, (20, 3))

    def test_umap_shape(self):
        """Tests dimensions of UMAP results"""
        out_umap = self.dut.umap(self.dummy_frames, do_fit_per_call=True)
        self.assertEqual(out_umap.shape, (20, 3))

    def test_export_and_load_state(self):
        """Tests saving as .joblib and loading a model using a tempfile."""
        out_original = self.dut.pca_full(self.dummy_frames, do_fit_per_call=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "test_model")

            self.dut.export_state(save_path)
            self.assertTrue(os.path.exists(save_path + ".joblib"))

            loaded_model = joblib.load(save_path + ".joblib")

            new_dut = FeatureExtraction(settings=self.set0)
            new_dut.load_state(loaded_model, fe_method_name="pca_full")

            out_loaded = new_dut.pca_full(self.dummy_frames, do_fit_per_call=False)

            np.testing.assert_array_almost_equal(out_original, out_loaded)

    def test_export_state_without_model(self):
        """Tests that exporting an untrained model raises a RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "fail_model")
            with self.assertRaisesRegex(RuntimeError, "No trained model available"):
                self.dut.export_state(save_path)

    def test_state_management_and_conflicts(self):
        """Tests state saving, erasing states, and conflict logic for all methods."""
        # initial state is empty
        self.assertFalse(self.dut.is_state_available)

        # pca_full
        self.dut.pca_full(self.dummy_frames, do_fit_per_call=False)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

        # try running pca_custom while pca_full is loaded
        with self.assertRaises(RuntimeError):
            self.dut.pca_custom(self.dummy_frames, svd_solver_mode="arpack", do_fit_per_call=False)

        # Erase state and verify
        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # pca_custom
        self.dut.pca_custom(self.dummy_frames, svd_solver_mode="arpack", do_fit_per_call=False)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_arpack")

        # try running umap while pca_custom is loaded
        with self.assertRaises(RuntimeError):
            self.dut.umap(self.dummy_frames, do_fit_per_call=False)

        # Erase state and verify
        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # umap
        self.dut.umap(self.dummy_frames, do_fit_per_call=False)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "umap")

        # try running pca_full while umap is loaded
        with self.assertRaises(RuntimeError):
            self.dut.pca_full(self.dummy_frames, do_fit_per_call=False)


if __name__ == "__main__":
    unittest.main()
