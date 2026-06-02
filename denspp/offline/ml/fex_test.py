import os
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from .fex import FeatureExtraction, SettingsFeature


class FeatExtractionTest(unittest.TestCase):
    def setUp(self):
        self.settings = SettingsFeature(num_features=3)
        self.dut = FeatureExtraction(settings=self.settings)

        # Create dummy data: 20 samples, 50 features
        # 20 samples because UMAP uses n_neighbors=15 by default
        rng = np.random.default_rng(42)
        self.dummy_frames = rng.random((20, 50))

    def test_pdac_min_max(self):
        """Test PDAC functions using one simple frame."""
        simple_frame = np.array([[2, 4, 1, 3, 5]])

        # PDAC MIN
        expected_min = np.array([[4, 6, 5, 1]])
        out_min = self.dut.pdac_min(simple_frame)
        np.testing.assert_array_equal(out_min, expected_min)

        # PDAC MAX
        expected_max = np.array([[10, 0, 5, 1]])
        out_max = self.dut.pdac_max(simple_frame)
        np.testing.assert_array_equal(out_max, expected_max)

    def test_pca_full_shape(self):
        """Test output shape of fit_transform_pca_full."""
        out_full = self.dut.fit_transform_pca_full(self.dummy_frames)
        self.assertEqual(out_full.shape, (20, 3))
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

    def test_pca_custom_shape(self):
        """Test output shape of fit_transform_pca_custom."""
        out_custom = self.dut.fit_transform_pca_custom(
            self.dummy_frames,
            svd_solver_mode="randomized",
            random_state=42,
        )
        self.assertEqual(out_custom.shape, (20, 3))
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_randomized")

    def test_umap_shape(self):
        """Test output shape of fit_transform_umap."""
        out_umap = self.dut.fit_transform_umap(self.dummy_frames)
        self.assertEqual(out_umap.shape, (20, 3))
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "umap")

    def test_ica_shape(self):
        """Test output shape of fit_transform_ica."""
        out_ica = self.dut.fit_transform_ica(self.dummy_frames)
        self.assertEqual(out_ica.shape, (20, 3))
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "ica")

    def test_transform_without_model(self):
        """Test that transform() raises if no fitted model is available."""
        with self.assertRaisesRegex(RuntimeError, "No fitted model available"):
            self.dut.transform(self.dummy_frames)

    def test_export_and_load_state(self):
        """Test saving/loading a model via joblib and reusing it through transform()."""
        # Fit and transform using PCA full
        out_original = self.dut.fit_transform_pca_full(self.dummy_frames)

        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "test_model")

            self.dut.export_state(save_path)
            self.assertTrue(os.path.exists(save_path + ".joblib"))

            loaded_model = joblib.load(save_path + ".joblib")

            new_dut = FeatureExtraction(settings=self.settings)
            new_dut.load_state(loaded_model, fe_method_name="pca_full")

            out_loaded = new_dut.transform(self.dummy_frames)

            np.testing.assert_array_almost_equal(out_original, out_loaded)

    def test_export_state_without_model(self):
        """Test that exporting without a fitted model raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "fail_model")
            with self.assertRaisesRegex(RuntimeError, "No fitted model available"):
                self.dut.export_state(save_path)

    def test_load_state_rejects_wrong_type(self):
        """Test that load_state() rejects unsupported model types."""
        with self.assertRaisesRegex(
            TypeError, "state_to_load must be a fitted PCA, UMAP, or FastICA model"
        ):
            self.dut.load_state("not_a_model", fe_method_name="pca_full")

    def test_load_state_rejects_unfitted_pca(self):
        """Test that load_state() rejects an unfitted PCA model."""
        unfitted_pca = PCA(n_components=3, svd_solver="full")

        with self.assertRaisesRegex(ValueError, "PCA state_to_load is not fitted"):
            self.dut.load_state(unfitted_pca, fe_method_name="pca_full")

    def test_erase_state(self):
        """Test that erase_state() clears the stored model and method."""
        self.dut.fit_pca_full(self.dummy_frames)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)
        self.assertIsNone(self.dut.fe_method)

    def test_load_state_conflict_with_existing_model(self):
        self.dut.fit_pca_full(self.dummy_frames)
        loaded_umap = UMAP().fit(self.dummy_frames)
        with self.assertRaises(RuntimeError):
            self.dut.load_state(loaded_umap, fe_method_name="umap")

    def test_state_management_and_conflicts(self):
        """Test state saving, erasing, and conflict logic for all trainable methods."""
        # Initial state
        self.assertFalse(self.dut.is_state_available)
        self.assertIsNone(self.dut.fe_method)

        # PCA full
        self.dut.fit_pca_full(self.dummy_frames)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

        # load PCA custom while PCA full is active
        with self.assertRaises(RuntimeError):
            self.dut.fit_pca_custom(self.dummy_frames, svd_solver_mode="arpack")

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # PCA custom
        self.dut.fit_pca_custom(self.dummy_frames, svd_solver_mode="arpack")
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_arpack")

        # load UMAP while PCA custom is active
        with self.assertRaises(RuntimeError):
            self.dut.fit_umap(self.dummy_frames)

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # UMAP
        self.dut.fit_umap(self.dummy_frames)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "umap")

        # loadICA while UMAP is active
        with self.assertRaises(RuntimeError):
            self.dut.fit_ica(self.dummy_frames)

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # ICA
        self.dut.fit_ica(self.dummy_frames)
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "ica")

        # load PCA full while ICA is active
        with self.assertRaises(RuntimeError):
            self.dut.fit_pca_full(self.dummy_frames)


if __name__ == "__main__":
    unittest.main()
