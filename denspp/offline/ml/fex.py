from dataclasses import dataclass
from typing import Literal

import joblib
import numpy as np
from sklearn.decomposition import PCA, FastICA
from umap import UMAP


@dataclass(frozen=True)
class SettingsFeature:
    """Configuration for feature extraction.

    Attributes:
        num_features: Number of output features / embedding dimensions.
    """

    num_features: int = 3
DefaultSettingsFeature = SettingsFeature(num_features=3)

class FeatureExtraction:
    """Feature extraction methods for signal frames."""

    def __init__(self, settings: SettingsFeature | None = None):
        self._settings = settings if settings is not None else SettingsFeature()
        self._model: PCA | FastICA | UMAP | None = None
        self.fe_method = None

    # ------------------------------------------------------------------
    # create estimators
    # ------------------------------------------------------------------
    def _create_pca(self, svd_solver: str, random_state: int | None) -> PCA:
        return PCA(
            n_components=self._settings.num_features,
            svd_solver=svd_solver,
            random_state=random_state if svd_solver == "randomized" else None,
        )

    def _create_umap(
        self, n_neighbors: int, random_state: int, min_dist: float
    ) -> UMAP:
        return UMAP(
            n_components=self._settings.num_features,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
        )

    def _create_ica(self, random_state: int) -> FastICA:
        return FastICA(
            n_components=self._settings.num_features, random_state=random_state
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @property
    def is_state_available(self) -> bool:
        return self._model is not None

    def _ensure_method_matches(self, expected_method: str) -> None:
        """Ensure that an already loaded/stored model matches the requested method."""
        if self._model is not None and self.fe_method != expected_method:
            raise RuntimeError(
                f"State conflict: expected '{expected_method}' but found '{self.fe_method}'. "
                "Please call erase_state() before using a different method."
            )

    def transform(self, frame_in: np.ndarray) -> np.ndarray:
        """Transform input data using the currently stored fitted model.

        Args:
            frame_in: Input array of shape (n_samples, n_features).

        Returns:
            Transformed data.

        Raises:
            RuntimeError: If no fitted model is available.
            TypeError: If the stored model does not provide a transform method.
        """
        if self._model is None:
            raise RuntimeError(
                "No fitted model available. Call a fit_* method first or load a fitted state."
            )

        if not hasattr(self._model, "transform"):
            raise TypeError(
                "The stored model does not provide a transform(...) method."
            )

        return self._model.transform(frame_in)

    def load_state(self, state_to_load, fe_method_name: str):
        """Load a fitted model into the class. If a different model has been fitted before, erase_state()
        needs to be called first.

        Args:
            state_to_load: A fitted model object providing transform(...).
            fe_method_name: Name of the feature extraction method the model belongs to.

        Raises:
            TypeError: If the model does not provide transform(...).
            ValueError: If the model is not fitted.
        """

        if not isinstance(state_to_load, (PCA, UMAP, FastICA)):
            raise TypeError(
                "state_to_load must be a fitted PCA, UMAP, or FastICA model."
            )

        self._ensure_method_matches(fe_method_name)

        if isinstance(state_to_load, PCA) and not hasattr(state_to_load, "components_"):
            raise ValueError("PCA state_to_load is not fitted (missing components_).")

        if isinstance(state_to_load, UMAP) and not hasattr(state_to_load, "embedding_"):
            raise ValueError("UMAP state_to_load is not fitted (missing embedding_).")

        if isinstance(state_to_load, FastICA) and not hasattr(
            state_to_load, "components_"
        ):
            raise ValueError("FastICA state_to_load is not fitted (missing components_).")

        self._model = state_to_load
        self.fe_method = fe_method_name

    def export_state(self, path2save: str):
        """Export the currently stored fitted model to a .joblib file.

        Args:
            path2save: Output path. If it does not already end with '.joblib',
                       the extension is added automatically.

        Raises:
            RuntimeError: If no fitted model is available.
        """
        if self._model is None:
            raise RuntimeError(
                "No fitted model available. Call a fit_* method first or load a fitted state."
            )

        if not path2save.endswith(".joblib"):
            path2save += ".joblib"

        joblib.dump(self._model, path2save)

    def erase_state(self):
        """Clear the currently stored model state."""
        self._model = None
        self.fe_method = None

    # ------------------------------------------------------------------
    # PDAC
    # ------------------------------------------------------------------
    def pdac_min(self, frame_in: np.ndarray) -> np.ndarray:
        """Compute PDAC features using the minimum value of each frame. Needs to be calculated freshly every time.

        Args:
            frame_in: Input array of shape (n_frames, frame_length).

        Returns:
            Array with features [a0, a1, ymax, ymin], where:
            - a0: sum(frame[:idx_valley] - ymin)
            - a1: sum(frame[idx_valley:] - ymin)
            - ymax: maximum frame value
            - ymin: minimum frame value
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
        """Compute PDAC features using the maximum value of each frame.

        Args:
            frame_in: Input array of shape (n_frames, frame_length).

        Returns:
            Array with features [a0, a1, ymax, ymin], where:
            - a0: sum(ymax - frame[:idx_peak])
            - a1: sum(ymax - frame[idx_peak:])
            - ymax: maximum frame value
            - ymin: minimum frame value
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

    # ------------------------------------------------------------------
    # PCA (full)
    # ------------------------------------------------------------------
    def fit_pca_full(self, frame_in: np.ndarray) -> None:
        """Fit a PCA model using the full SVD solver and store it internally."""
        self._ensure_method_matches("pca_full")
        self._model = self._create_pca("full", None)
        self._model.fit(frame_in)
        self.fe_method = "pca_full"

    def fit_transform_pca_full(self, frame_in: np.ndarray) -> np.ndarray:
        """Fit a PCA model using the full SVD solver, store it, and return transformed data."""
        self._ensure_method_matches("pca_full")
        self._model = self._create_pca("full", None)
        self.fe_method = "pca_full"
        return self._model.fit_transform(frame_in)

    # ------------------------------------------------------------------
    # PCA (custom)
    # ------------------------------------------------------------------
    def fit_pca_custom(
        self,
        frame_in: np.ndarray,
        svd_solver_mode: Literal["covariance_eigh", "arpack", "randomized", "full"],
        random_state: int | None = None,
    ) -> None:
        """Fit a PCA model with a selectable SVD solver and store it internally."""
        method_name = f"pca_{svd_solver_mode}"
        self._ensure_method_matches(method_name)

        self._model = self._create_pca(svd_solver_mode, random_state)
        self._model.fit(frame_in)
        self.fe_method = method_name

    def fit_transform_pca_custom(
        self,
        frame_in: np.ndarray,
        svd_solver_mode: Literal["covariance_eigh", "arpack", "randomized", "full"],
        random_state: int | None = None,
    ) -> np.ndarray:
        """Fit a PCA model with a selectable SVD solver, store it, and return transformed data."""
        method_name = f"pca_{svd_solver_mode}"
        self._ensure_method_matches(method_name)

        self._model = self._create_pca(svd_solver_mode, random_state)
        self.fe_method = method_name
        return self._model.fit_transform(frame_in)

    # ------------------------------------------------------------------
    # UMAP
    # ------------------------------------------------------------------
    def fit_umap(
        self,
        frame_in: np.ndarray,
        n_neighbors: int = 15,
        random_state: int = 42,
        min_dist: float = 0.1,
    ) -> None:
        """Fit a UMAP model and store it internally."""
        self._ensure_method_matches("umap")
        self._model = self._create_umap(n_neighbors, random_state, min_dist)
        self._model.fit(frame_in)
        self.fe_method = "umap"

    def fit_transform_umap(
        self,
        frame_in: np.ndarray,
        n_neighbors: int = 15,
        random_state: int = 42,
        min_dist: float = 0.1,
    ) -> np.ndarray:
        """Fit a UMAP model, store it, and return transformed data."""
        self._ensure_method_matches("umap")
        self._model = self._create_umap(n_neighbors, random_state, min_dist)
        self.fe_method = "umap"
        return self._model.fit_transform(frame_in)

    # ------------------------------------------------------------------
    # ICA
    # ------------------------------------------------------------------
    def fit_ica(self, frame_in: np.ndarray, random_state: int = 42) -> None:
        """Fit a FastICA model and store it internally."""
        self._ensure_method_matches("ica")
        self._model = self._create_ica(random_state)
        self._model.fit(frame_in)
        self.fe_method = "ica"

    def fit_transform_ica(
        self, frame_in: np.ndarray, random_state: int = 42
    ) -> np.ndarray:
        """Fit a FastICA model, store it, and return transformed data."""
        self._ensure_method_matches("ica")
        self._model = self._create_ica(random_state)
        self.fe_method = "ica"
        return self._model.fit_transform(frame_in)