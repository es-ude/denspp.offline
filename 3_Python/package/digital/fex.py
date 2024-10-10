import dataclasses
import numpy as np
from sklearn.decomposition import PCA
from torch import load, from_numpy


@dataclasses.dataclass
class SettingsFeature:
    """"Individuall data class to configure feature extractor and cluster"""
    no_features: int


RecommendedSettingsFeature = SettingsFeature(
    no_features=3
)


class FeatureExtraction:
    """Class with functions for feature extraction"""
    def __init__(self, setting: SettingsFeature):
        self.settings = setting

    def pdac_min(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with minimum value on spike frames"""
        pdac_out = []
        for idx, frame in enumerate(frame_in):
            # Parameter berechnen
            ymin = np.min(frame)
            ymax = np.max(frame)
            xmin = np.where(frame == ymin)
            a0 = np.sum(frame[0:xmin[0][0]] - ymin)
            a1 = np.sum(frame[xmin[0][0]:-1] - ymin)
            # Akkumulation
            pdac = [a0, a1, ymax, ymin]
            pdac_out.append(pdac)
        return np.array(pdac_out)

    def pdac_max(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with maximum value on spike frames"""
        pdac_out = []
        for idx, frame in enumerate(frame_in):
            # Parameter berechnen
            ymin = np.min(frame)
            ymax = np.max(frame)
            xmax = np.where(frame == ymax)
            a0 = np.sum(ymax - frame[0:xmax[0][0]])
            a1 = np.sum(ymax - frame[xmax[0][0]:-1])
            # Akkumulation
            pdac = [a0, a1, ymax, ymin]
            pdac_out.append(pdac)
        return np.array(pdac_out)

    def pca(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing Principial Component Analysis (PCA) on spike frames"""
        frame_pca = np.transpose(frame_in)
        pca = PCA(
            n_components=self.settings.no_features,
            svd_solver="full"
        )
        pca.fit(frame_pca)
        feat0 = pca.components_
        features = np.transpose(feat0)
        return features

    def autoencoder(self, frame_in: np.ndarray, path2model: str) -> np.ndarray:
        """Using autoencoder for feature extraction"""
        model_ae = load(path2model)
        model_ae = model_ae.to("cpu")
        feat = model_ae(from_numpy(np.array(frame_in, dtype=np.float32)))[0]
        return feat.detach().numpy()
