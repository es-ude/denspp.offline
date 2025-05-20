from dataclasses import dataclass
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


@dataclass
class ClusterSettings:
    n_channels: int = 60
    eps: float = None  # Wird dynamisch gesetzt
    min_samples: int = 5
    denoise: bool = True
    min_cluster_size: int = 10
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

        self.data_file_name = "../data/A1R1a_light_stim_pre_spike_dictionary.npy"
        self.project_base_dir = Path(__file__).resolve().parent.parent
        self.data_dir_path = self.project_base_dir / settings.data_subdir
        self.file_path = self.data_dir_path / self.data_file_name

        self.results = []
        self.signal_dict = {}
        self.min_cluster_size = settings.min_cluster_size
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
        # Ensure n_neighbors does not exceed the number of samples
        n_neighbors = min(self.min_samples, len(features))
        neighbors = NearestNeighbors(n_neighbors=n_neighbors)

        neighbors_fit = neighbors.fit(features)
        distances, _ = neighbors_fit.kneighbors(features)
        k_distances = np.sort(distances[:, -1])

        # Optional: Automatically suggest eps based on the 90th percentile
        eps_suggestion = np.percentile(k_distances, 90)
        print(f"Automatically suggested eps value: {eps_suggestion:.3f}")
        return eps_suggestion


        # Optional: automatischer Vorschlag durch z. B. 90%-Perzentil
        eps_suggestion = np.percentile(k_distances, 90)
        print(f"Automatisch vorgeschlagener eps-Wert: {eps_suggestion:.3f}")
        return eps_suggestion

    def fit(self, data):
        frames_all = data['aligned_frames_list']
        positions_all = data['spike_indices_list']
        timestamps = np.array(data['time'])

        for ch in range(self.n_channels - 1):  # Für jeden Kanal
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

            # Ground-Truth-Labels extrahieren
            ground_truth = self.signal_dict["feature"][positions]

            entry = {
                'channel': ch,
                'labels': labels,
                'features': features,
                'times': times,
                'positions': positions,
                'ground_truth': ground_truth  # Ground-Truth-Daten speichern
            }
            self._filter_noise_clusters(entry)
            self.results.append(entry)  # Ergebnisse pro Kanal hinzufügen


    def get_results(self):
        return self.results

    def plot_interactive_3d_clusters(self, n_channels=60):
        noise_label = -1
        for entry in self.results[:n_channels]:
            channel = entry['channel']
            labels = entry['labels']
            features = entry['feature']

            # Entfernen aller Noise-Punkte
            valid_indices = labels != noise_label
            filtered_features = features[valid_indices]
            filtered_labels = labels[valid_indices]

            # Plot nur mit validierten Cluster-Daten
            fig = px.scatter_3d(
                x=filtered_features[:, 0],
                y=filtered_features[:, 1],
                z=filtered_features[:, 2],
                color=[f"Cluster {label}" for label in filtered_labels],
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

    def _filter_noise_clusters(self, entry):
        labels = entry['labels']
        unique_labels, counts = np.unique(labels, return_counts=True)
        small_clusters = {label for label, count in zip(unique_labels, counts) if count < self.min_cluster_size and label != -1}

        entry['labels'] = np.array([
            -1 if label in small_clusters else label
            for label in labels
        ])

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    def evaluate_with_confusion_matrix(self, results, save_fig=True, output_dir="figures"):
        """
        Erzeuge wissenschaftliche Confusion-Matrix-Plots für jeden Kanal.
        Optional: Speichern als .eps im angegebenen Verzeichnis.
        """

        for i, entry in enumerate(results):
            labels = entry['labels']  # Clustering-Labels
            ground_truth = entry['ground_truth']  # Ground-Truth-Labels

            if ground_truth is not None and i == 8:
                # Entferne Noise-Punkte (Label -1)
                valid_indices = labels != -1
                filtered_labels = labels[valid_indices]
                filtered_ground_truth = ground_truth[valid_indices]

                # Confusion Matrix berechnen
                matrix = confusion_matrix(filtered_ground_truth, filtered_labels)
                print(f"Confusion Matrix für Kanal {entry['channel']}:\n", matrix)

                # Plot
                plt.figure(figsize=(6, 5))
                sns.set(font_scale=1.2)
                ax = sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar_kws={"label": "Anzahl"})
                ax.set_title(f"Confusion Matrix – Kanal {entry['channel']}", fontsize=14)
                ax.set_xlabel("Cluster-Labels", fontsize=12)
                ax.set_ylabel("Ground-Truth-Labels", fontsize=12)
                plt.tight_layout()

                if save_fig:
                    filename = f"confusion_matrix_channel_{entry['channel']}.eps"
                    plt.savefig(filename, format='eps', dpi=300)
                    print(f"Gespeichert als: {filename}")

                plt.show()

            else:
                print(f"Keine Ground-Truth-Labels für Kanal {entry['channel']} verfügbar.")


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
        # Confusion Matrix Evaluation
        print("\n#### Clustering-Evaluation mit Confusion Matrix ####\n")
        clusterer.evaluate_with_confusion_matrix(results)

        # Interaktive Visualisierung der Cluster (Optional)
        clusterer.plot_interactive_3d_clusters(n_channels=10)
    else:
        print("Keine Clustering-Ergebnisse vorhanden.")

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


