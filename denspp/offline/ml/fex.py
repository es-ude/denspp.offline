from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.decomposition import PCA, FastICA
from umap import UMAP


@dataclass
class SettingsFeature:
    """ Individual data class to configure feature extractor and cluster
    Attributes:
        num_features:   Number of features /  dimensions of feature space"""

    num_features: int


DefaultSettingsFeature = SettingsFeature(num_features=3)


class FeatureExtraction:
    """Class containing feature extraction methods for signal frames."""

    def __init__(self, settings: SettingsFeature = DefaultSettingsFeature):
        self._settings = settings
        self._feat_result = None
        self.fe_method = None

    @property
    def is_state_available(self) -> bool:
        return self._feat_result is not None

    def load_state(self, state_to_load, fe_method_name: str):
        """Loads a pretrained model (e.g., PCA or UMAP) into the class.

        :param state_to_load: The pretrained model object.
        :param method_name: The name of the method the state belongs to (e.g. 'umap', 'pca_full').
        :raises TypeError: If the model lacks a transform method.
        :raises ValueError: If the model is not properly fitted.
        """

        if not hasattr(state_to_load, "transform"):
            raise TypeError(
                "state_to_load must provide a .transform(...) method (e.g., fitted PCA or fitted UMAP)."
            )

        if isinstance(state_to_load, PCA) and not hasattr(state_to_load, "components_"):
            raise ValueError("PCA state_to_load is not fitted (missing components_).")

        if isinstance(state_to_load, UMAP) and not hasattr(state_to_load, "embedding_"):
            raise ValueError("UMAP state_to_load is not fitted (missing embedding_).")

        if isinstance(state_to_load, FastICA) and not hasattr(state_to_load, "components_"):
            raise ValueError("ICA state_to_load is not fitted (missing components_).")

        self._feat_result = state_to_load
        self.fe_method = fe_method_name

    def export_state(self, path2save: str):
        """Exports the trained model state to a .joblib file.

        :param path2save: Path and filename (without extension).
        :raises RuntimeError: If no model is currently trained.
        """
        if self._feat_result is None:
            raise RuntimeError(
                "No trained model available. Please run a trainable feature extraction method first e.g. umap or pca"
                " (do_fit_per_call must be False)"
            )

        import joblib

        model = self._feat_result
        joblib.dump(model, path2save + ".joblib")

    def erase_state(self):
        """Clears the currently stored model state."""
        self._feat_result = None
        self.fe_method = None

    def pdac_min(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with minimum value on frames
        :param frame_in:    Numpy array with input frames
        :return:            Numpy array with features [sum_until_xmin, sum_after_xmin, ymax, ymin]
        """
        pdac_out = []
        for frame in frame_in:
            ymin = np.min(frame)
            idx_valley = np.argmin(frame)
            ymax = np.max(frame)
            a0 = np.sum(frame[:idx_valley] - ymin)
            a1 = np.sum(frame[idx_valley:] - ymin)
            pdac_out.append([a0, a1, ymax, ymin])
        return np.array(pdac_out)

    def pdac_max(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with maximum value on frames
        :param frame_in:    Numpy array with input frames
        :return:            Numpy array with features [sum_until_xmax, sum_after_xmax, ymax, ymin]
        """
        pdac_out = []
        for frame in frame_in:
            ymin = np.min(frame)
            ymax = np.max(frame)
            idx_peak = np.argmax(frame)
            a0 = np.sum(ymax - frame[:idx_peak])
            a1 = np.sum(ymax - frame[idx_peak:])
            pdac_out.append([a0, a1, ymax, ymin])
        return np.array(pdac_out)

    def pca_full(
        self,
        frame_in: np.ndarray,
        do_fit_per_call: bool = True,
    ) -> np.ndarray:
        """Performs PCA with a full SVD solver.
        :param frame_in: Numpy array with input frames.
        :param do_fit_per_call: If True, refits model. If False, reuses existing state.
        :return: PCA-transformed features.
        """
        if self._feat_result is not None and self.fe_method != "pca_full":
            raise RuntimeError(
                f"State conflict: Expected 'pca_full' but found '{self.fe_method}'. \n"
                "Please call 'erase_state first before running a different method"
            )

        self.fe_method = "pca_full"
        if do_fit_per_call:
            pca = PCA(n_components=self._settings.num_features, svd_solver="full")
            return pca.fit_transform(frame_in)

        else:
            if self._feat_result is None:
                self._feat_result = PCA(n_components=self._settings.num_features, svd_solver="full")
                return self._feat_result.fit_transform(frame_in)
            return self._feat_result.transform(frame_in)

    def pca_custom(
        self,
        frame_in: np.ndarray,
        svd_solver_mode: Literal["covariance_eigh", "arpack", "randomized", "full"],
        do_fit_per_call: bool = True,
        random_state: int | None = None,
    ) -> np.ndarray:
        """Performs PCA with a custom selectable SVD solver.

        :param frame_in: Numpy array with input frames.
        :param svd_solver_mode: Solver algorithm to use.
        :param do_fit_per_call: If True, refits model. If False, reuses existing state.
        :param random_state: Seed for randomized solver.
        :return: PCA-transformed features.
        """

        if self._feat_result is not None and self.fe_method != f"pca_{svd_solver_mode}":
            raise RuntimeError(
                f"State conflict: Expected 'pca_{svd_solver_mode}' but found '{self.fe_method}'. \n"
                "Please call 'erase_state first before running a different method"
            )

        self.fe_method = f"pca_{svd_solver_mode}"

        if do_fit_per_call:
            pca = PCA(
                n_components=self._settings.num_features,
                svd_solver=svd_solver_mode,
                random_state=random_state if svd_solver_mode == "randomized" else None,
            )
            return pca.fit_transform(frame_in)

        else:
            if self._feat_result is None:
                self._feat_result = PCA(
                    n_components=self._settings.num_features,
                    svd_solver=svd_solver_mode,
                    random_state=random_state if svd_solver_mode == "randomized" else None,
                )
                return self._feat_result.fit_transform(frame_in)
            return self._feat_result.transform(frame_in)

    def umap(
        self,
        frame_in: np.ndarray,
        n_neighbors: int = 15,
        random_state: int = 42,
        min_dist: float = 0.1,
        do_fit_per_call: bool = True,
    ) -> np.ndarray:
        """Performing UMAP (Uniform Manifold Approximation and Projection) on spike frames
        :param frame_in:         Numpy array with input frames
        :param n_neighbors:            Size of local neighborhood for manifold approximation
        :param random_state:     Integer for reproducible UMAP results
        :param min_dist:               Minimum distance between embedded points
        :param do_fit_per_call:    If True, refits model every call. If False, reuses existing state.
        :return:                 Numpy array with N features
        """
        if self._feat_result is not None and self.fe_method != "umap":
            raise RuntimeError(
                f"State conflict: Expected 'umap' but found '{self.fe_method}'. \n"
                "Please call 'erase_state first before running a different method"
            )
        self.fe_method = "umap"
        if do_fit_per_call:
            reducer = UMAP(
                n_components=self._settings.num_features,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
            return reducer.fit_transform(frame_in)

        else:
            if self._feat_result is None:
                self._feat_result = UMAP(
                    n_components=self._settings.num_features,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=random_state,
                )
                return self._feat_result.fit_transform(frame_in)
            return self._feat_result.transform(frame_in)

    def ica(
        self, frame_in: np.ndarray, random_state: int = 42, do_fit_per_call: bool = True
    ) -> np.ndarray:
        """Performing Independent Component Analysis (ICA) on spike frames
        :param frame_in:         Numpy array with input frames
        :param random_state:     Integer for reproducible ICA results
        :param do_fit_per_call:     If True, refits model every call. If False, reuses existing state.
        :return:                 Numpy array with N features
        """
        if self._feat_result is not None and self.fe_method != "ica":
            raise RuntimeError(
                f"State conflict: Expected 'ica' but found '{self.fe_method}'. \n"
                "Please call 'erase_state first before running a different method"
            )
        self.fe_method = "ica"
        if do_fit_per_call:
            model_ica = FastICA(n_components=self._settings.num_features, random_state=random_state)
            return model_ica.fit_transform(frame_in)

        else:
            if self._feat_result is None:  # 1st Iteration
                self._feat_result = FastICA(
                    n_components=self._settings.num_features, random_state=random_state
                )
                return self._feat_result.fit_transform(frame_in)
            return self._feat_result.transform(frame_in)
