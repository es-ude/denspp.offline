import os
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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

    def test_tsne_shape(self):
        """Test output shape of fit_transform_tsne."""
        out_tsne = self.dut.fit_transform_tsne(
            self.dummy_frames,
            perplexity=5.0,
            random_state=42,
        )

        self.assertEqual(out_tsne.shape, (20, 3))
        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "tsne")

    def test_tsne_fit_state(self):
        """Test that fit_tsne stores a fitted t-SNE model."""
        self.dut.fit_tsne(
            self.dummy_frames,
            perplexity=5.0,
            random_state=42,
        )

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "tsne")
        self.assertIsInstance(self.dut._model, TSNE)
        self.assertTrue(hasattr(self.dut._model, "embedding_"))
        self.assertEqual(self.dut._model.embedding_.shape, (20, 3))

    def test_tsne_auto_uses_barnes_hut_for_three_dimensions(self):
        """Test that auto selects Barnes-Hut for fewer than four dimensions."""
        self.dut.fit_tsne(
            self.dummy_frames,
            perplexity=5.0,
            method="auto",
        )

        self.assertEqual(self.dut._model.method, "barnes_hut")

    def test_tsne_auto_uses_exact_for_four_dimensions(self):
        """Test that auto selects exact t-SNE for four-dimensional output."""
        settings = SettingsFeature(num_features=4)
        dut = FeatureExtraction(settings=settings)

        out_tsne = dut.fit_transform_tsne(
            self.dummy_frames,
            perplexity=5.0,
            method="auto",
        )

        self.assertEqual(out_tsne.shape, (20, 4))
        self.assertEqual(dut._model.method, "exact")
        self.assertEqual(dut.fe_method, "tsne")

    def test_tsne_rejects_barnes_hut_for_four_dimensions(self):
        """Test that Barnes-Hut rejects four-dimensional output."""
        settings = SettingsFeature(num_features=4)
        dut = FeatureExtraction(settings=settings)

        with self.assertRaisesRegex(
            ValueError,
            "does not support n_components=4",
        ):
            dut.fit_tsne(
                self.dummy_frames,
                perplexity=5.0,
                method="barnes_hut",
            )

    def test_tsne_transform_is_not_supported(self):
        """Test that transform is unavailable for a fitted t-SNE model."""
        self.dut.fit_tsne(
            self.dummy_frames,
            perplexity=5.0,
        )

        with self.assertRaisesRegex(
            TypeError,
            r"does not provide a transform\(\.\.\.\) method",
        ):
            self.dut.transform(self.dummy_frames)

    def test_transform_without_model(self):
        """Test that transform raises if no fitted model is available."""
        with self.assertRaisesRegex(RuntimeError, "No fitted model available"):
            self.dut.transform(self.dummy_frames)

    def test_export_and_load_state(self):
        """Test saving/loading a model via joblib and reusing it through transform."""
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

    def test_export_and_load_tsne_state(self):
        """Test saving and loading a fitted t-SNE model via joblib."""
        out_original = self.dut.fit_transform_tsne(
            self.dummy_frames,
            perplexity=5.0,
            random_state=42,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "test_tsne_model")

            self.dut.export_state(save_path)
            self.assertTrue(os.path.exists(save_path + ".joblib"))

            loaded_model = joblib.load(save_path + ".joblib")

            new_dut = FeatureExtraction(settings=self.settings)
            new_dut.load_state(loaded_model, fe_method_name="tsne")

            self.assertTrue(new_dut.is_state_available)
            self.assertEqual(new_dut.fe_method, "tsne")
            self.assertIsInstance(new_dut._model, TSNE)

            np.testing.assert_array_almost_equal(
                out_original,
                new_dut._model.embedding_,
            )

    def test_export_state_without_model(self):
        """Test that exporting without a fitted model raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "fail_model")

            with self.assertRaisesRegex(RuntimeError, "No fitted model available"):
                self.dut.export_state(save_path)

    def test_load_state_rejects_wrong_type(self):
        """Test that load_state rejects unsupported model types."""
        with self.assertRaisesRegex(
            TypeError,
            "state_to_load must be a fitted PCA, UMAP, or FastICA model",
        ):
            self.dut.load_state("not_a_model", fe_method_name="pca_full")

    def test_load_state_rejects_unfitted_pca(self):
        """Test that load_state rejects an unfitted PCA model."""
        unfitted_pca = PCA(n_components=3, svd_solver="full")

        with self.assertRaisesRegex(
            ValueError,
            "PCA state_to_load is not fitted",
        ):
            self.dut.load_state(unfitted_pca, fe_method_name="pca_full")

    def test_load_state_rejects_unfitted_tsne(self):
        """Test that load_state rejects an unfitted t-SNE model."""
        unfitted_tsne = TSNE(
            n_components=3,
            perplexity=5.0,
            random_state=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "TSNE state_to_load is not fitted",
        ):
            self.dut.load_state(unfitted_tsne, fe_method_name="tsne")

    def test_erase_state(self):
        """Test that erase_state clears the stored model and method."""
        self.dut.fit_pca_full(self.dummy_frames)

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

        self.dut.erase_state()

        self.assertFalse(self.dut.is_state_available)
        self.assertIsNone(self.dut.fe_method)

    def test_erase_tsne_state(self):
        """Test that erase_state clears a fitted t-SNE model."""
        self.dut.fit_tsne(
            self.dummy_frames,
            perplexity=5.0,
        )

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "tsne")

        self.dut.erase_state()

        self.assertFalse(self.dut.is_state_available)
        self.assertIsNone(self.dut.fe_method)

    def test_load_state_conflict_with_existing_model(self):
        self.dut.fit_pca_full(self.dummy_frames)
        loaded_umap = UMAP().fit(self.dummy_frames)

        with self.assertRaises(RuntimeError):
            self.dut.load_state(loaded_umap, fe_method_name="umap")

    def test_load_tsne_state_conflict_with_existing_model(self):
        """Test that a t-SNE model cannot replace a different active state."""
        self.dut.fit_pca_full(self.dummy_frames)

        loaded_tsne = TSNE(
            n_components=3,
            perplexity=5.0,
            random_state=42,
        ).fit(self.dummy_frames)

        with self.assertRaises(RuntimeError):
            self.dut.load_state(loaded_tsne, fe_method_name="tsne")

    def test_state_management_and_conflicts(self):
        """Test state saving, erasing, and conflict logic for all trainable methods."""
        self.assertFalse(self.dut.is_state_available)
        self.assertIsNone(self.dut.fe_method)

        # PCA full
        self.dut.fit_pca_full(self.dummy_frames)

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_full")

        with self.assertRaises(RuntimeError):
            self.dut.fit_pca_custom(
                self.dummy_frames,
                svd_solver_mode="arpack",
            )

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # PCA custom
        self.dut.fit_pca_custom(
            self.dummy_frames,
            svd_solver_mode="arpack",
        )

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "pca_arpack")

        with self.assertRaises(RuntimeError):
            self.dut.fit_umap(self.dummy_frames)

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # UMAP
        self.dut.fit_umap(self.dummy_frames)

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "umap")

        with self.assertRaises(RuntimeError):
            self.dut.fit_ica(self.dummy_frames)

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # ICA
        self.dut.fit_ica(self.dummy_frames)

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "ica")

        with self.assertRaises(RuntimeError):
            self.dut.fit_tsne(
                self.dummy_frames,
                perplexity=5.0,
            )

        self.dut.erase_state()
        self.assertFalse(self.dut.is_state_available)

        # t-SNE
        self.dut.fit_tsne(
            self.dummy_frames,
            perplexity=5.0,
        )

        self.assertTrue(self.dut.is_state_available)
        self.assertEqual(self.dut.fe_method, "tsne")

        with self.assertRaises(RuntimeError):
            self.dut.fit_pca_full(self.dummy_frames)


if __name__ == "__main__":
    unittest.main()