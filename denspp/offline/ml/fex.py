from dataclasses import dataclass
import numpy as np
from sklearn.decomposition import PCA
from torch import load, from_numpy


@dataclass
class SettingsFeature:
    """"Individual data class to configure feature extractor and cluster"""
    pass


DefaultSettingsFeature = SettingsFeature()


class FeatureExtraction:
    """Class with functions for feature extraction"""
    def __init__(self, settings: SettingsFeature=DefaultSettingsFeature):
        self.settings = settings

    def pdac_min(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with minimum value on frames
        :param frame_in:    Numpy array with input frames
        :return:            Numpy array with features [sum_until_xmin, sum_after_xmin, ymax, ymin]
        """
        pdac_out = []
        for idx, frame in enumerate(frame_in):
            ymin = np.min(frame)
            ymax = np.max(frame)
            xmin = np.where(frame == ymin)
            a0 = np.sum(frame[0:xmin[0][0]] - ymin)
            a1 = np.sum(frame[xmin[0][0]:-1] - ymin)
            pdac = [a0, a1, ymax, ymin]
            pdac_out.append(pdac)
        return np.array(pdac_out)

    def pdac_max(self, frame_in: np.ndarray) -> np.ndarray:
        """Performing the Peak Detection with Area Computation (PDAC) method with maximum value on frames
        :param frame_in:    Numpy array with input frames
        :return:            Numpy array with features [sum_until_xmax, sum_after_xmax, ymax, ymin]
        """
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

    def pca(self, frame_in: np.ndarray, num_features: int) -> np.ndarray:
        """Performing Principial Component Analysis (PCA) on spike frames
        :param frame_in:        Numpy array with input frames
        :param num_features:    Number of features to extract
        :return:                Numpy array with N features
        """
        frame_pca = np.transpose(frame_in)
        pca = PCA(
            n_components=num_features,
            svd_solver="full"
        )
        pca.fit(frame_pca)
        feat0 = pca.components_
        features = np.transpose(feat0)
        return features

    def autoencoder(self, frame_in: np.ndarray, path2model: str) -> np.ndarray:
        """Using autoencoder for feature extraction
        :param frame_in:        Numpy array with input frames
        :param path2model:      Path to saved model
        :return:                Numpy array with N features
        """
        model_ae = load(path2model).to("cpu")
        feat = model_ae(from_numpy(np.array(frame_in, dtype=np.float32)))[0]
        return feat.detach().numpy()
