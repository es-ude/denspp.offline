from dataclasses import dataclass
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter
import plotly.express as px
import matplotlib.pyplot as plt


@dataclass
class ClusterSettings:
    n_channels: int = 60
    eps: float = None  # Wird dynamisch gesetzt
    min_samples: int = 5
    denoise: bool = True
    window_length: int = 11
    polyorder: int = 2
    n_pca_components: int = 3
    outlier_z_thresh: float = 4.0
    data_subdir: str = "data"


class MultichannelSpikeClustering:
    def __init__(self, settings: ClusterSettings):
        self.settings = settings
        self.n_channels = settings.n_channels
        self.min_samples = settings.min_samples
        self.denoise = settings.denoise
        self.window_length = settings.window_length
        self.polyorder = settings.polyorder
        self.n_pca_components = settings.n_pca_components
        self.outlier_z_thresh = settings.outlier_z_thresh

        self.data_file_name = "../data/A1R1a_ASIC_1S_1000_15_spike_dictionary.npy"
        self.project_base_dir = Path(__file__).resolve().parent.parent
        self.data_dir_path = self.project_base_dir / settings.data_subdir
        self.file_path = self.data_dir_path / self.data_file_name

        self.results = []
        self.signal_dict = {}

    def load_file_as_dict(self):
        if not self.file_path.exists():
            print(f"Fehler: Datei wurde nicht gefunden: {self.file_path}")
            return
        try:
            self.signal_dict = np.load(self.file_path, allow_pickle=True).item()
            print(f".npy-Datei erfolgreich als Dictionary geladen: {self.file_path}")
        except Exception as e:
            print(f"Fehler beim Laden der Datei: {self.file_path}. Details: {e}")

    def _denoise_spikes(self, spikes):
        return np.array([
            savgol_filter(spike, self.window_length, self.polyorder)
            for spike in spikes
        ])

    def _normalize(self, spikes):
        return StandardScaler().fit_transform(spikes)

    def _pca_reduce(self, spikes):
        return PCA(n_components=self.n_pca_components).fit_transform(spikes)

    def _remove_outliers(self, features):
        z = np.abs((features - features.mean(axis=0)) / features.std(axis=0))
        mask = (z < self.outlier_z_thresh).all(axis=1)
        return features[mask], mask

    def _estimate_eps(self, features):
        neighbors = NearestNeighbors(n_neighbors=self.min_samples)
        neighbors_fit = neighbors.fit(features)
        distances, _ = neighbors_fit.kneighbors(features)
        k_distances = np.sort(distances[:, -1])

        plt.figure(figsize=(8, 4))
        plt.plot(k_distances)
        plt.title(f"k-Distanz-Plot (k = {self.min_samples})")
        plt.xlabel("Punkte (sortiert)")
        plt.ylabel(f"Distanz zum {self.min_samples}. Nachbarn")
        plt.grid(True)
        plt.show()

        # Optional: automatischer Vorschlag durch z. B. 90%-Perzentil
        eps_suggestion = np.percentile(k_distances, 90)
        print(f"Automatisch vorgeschlagener eps-Wert: {eps_suggestion:.3f}")
        return eps_suggestion

    def fit(self, data):
        frames_all = data['aligned_frames_list']
        positions_all = data['spike_indices_list']
        timestamps = np.array(data['time'])

        for ch in range(self.n_channels-1):
            spikes = np.array(frames_all[ch])
            positions = np.array(positions_all[ch])

            if len(spikes) == 0 or len(positions) == 0:
                continue

            if self.denoise:
                spikes = self._denoise_spikes(spikes)

            spikes = self._normalize(spikes)
            features = self._pca_reduce(spikes)
            features, mask = self._remove_outliers(features)

            if self.settings.eps is None:
                eps = self._estimate_eps(features)
            else:
                eps = self.settings.eps

            labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit_predict(features)
            positions = positions[mask]
            times = timestamps[positions]

            self.results.append({
                'channel': ch,
                'labels': labels,
                'features': features,
                'times': times,
                'positions': positions
            })

    def get_results(self):
        return self.results

    def plot_interactive_3d_clusters(self, n_channels=1):
        noise_label = -1
        for entry in self.results[:n_channels]:
            channel = entry['channel']
            labels = entry['labels']
            features = entry['features']

            fig = px.scatter_3d(
                x=features[:, 0],
                y=features[:, 1],
                z=features[:, 2],
                color=[f"Cluster {label}" if label != noise_label else "Noise" for label in labels],
                title=f"3D PCA-Scatterplot für Kanal {channel}",
                labels={"color": "Cluster"}
            )

            fig.update_layout(
                scene=dict(
                    xaxis_title="PCA-Komponente 1",
                    yaxis_title="PCA-Komponente 2",
                    zaxis_title="PCA-Komponente 3"
                )
            )
            fig.show()


if __name__ == "__main__":
    settings = ClusterSettings(
        n_channels=60,
        denoise=True,
        window_length=11,
        polyorder=2,
        n_pca_components=3,
        outlier_z_thresh=4.0,
        data_subdir="data"
    )
    clusterer = MultichannelSpikeClustering(settings)
    clusterer.load_file_as_dict()
    clusterer.fit(clusterer.signal_dict)
    results = clusterer.get_results()

    if results:
        clusterer.plot_interactive_3d_clusters(n_channels=1)
    else:
        print("Keine Clustering-Ergebnisse vorhanden.")
