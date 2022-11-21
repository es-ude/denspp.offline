import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from settings import Settings

class FEC:
    def __init__(self, setting: Settings):
        self.realtime_mode = setting.realtime_mode

        # --- Settings for feature extraction
        self.mode_feature = setting.mode_feature

        # --- Settings for clustering
        self.mode_cluster = setting.mode_cluster
        self.no_cluster = setting.no_cluster

    def fe_pda(self, frame_in: np.ndarray):
        # TODO: Methode PDA einfügen
        ...

    def fe_pca(self, frame_in: np.ndarray):
        if self.realtime_mode:
            ...
        else:
            frame_pca = np.rot90(frame_in, k=1)
            pca = PCA(n_components=10, svd_solver="full")
            pca.fit(frame_pca)
            features = pca.components_
            precision = pca.get_precision()
            score = pca.score(frame_pca)
            print("... mean value of precision:", np.mean(precision))
            print("... score of feature extraction:", score)

        return features

    def clustering(self, features: np.ndarray):
        #for runs in range(0, 9):
        feat_in = np.rot90(features, k=1)

        cluster = KMeans(
            init="random",
            n_clusters=3,
            n_init=1, max_iter=100
        )
        cluster.fit(feat_in, sample_weight=None)
        results = cluster.labels_
        sse = cluster.inertia_

        return results, sse

    def calc_spiketicks(self, uin, xpos, cluster):
        no_cluster = np.unique(cluster)
        ticks = np.zeros(shape=(no_cluster.size, uin.size), dtype=int)

        idx = 0
        for val in xpos:
            # print("#:", idx, "- Pos:", val, "- ID:", cluster[idx])
            ticks[cluster[idx], val] = 1
            idx += 1
        A = 1

        return ticks