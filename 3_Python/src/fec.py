import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from settings import Settings

class FEC:
    def __init__(self, setting: Settings):
        self.__no_cluster = setting.no_cluster

    def fe_pdac(self, frame_in: np.ndarray):
        # TODO: Methode PDAC einfügen
        pass

    def fe_pca(self, frame_in: np.ndarray):
        frame_pca = np.transpose(frame_in)
        pca = PCA(
            n_components=2,
            svd_solver="full"
        )
        pca.fit(frame_pca)
        feat0 = pca.components_
        #precision = pca.get_precision()
        #score = pca.score(frame_pca)
        #print("... mean value of precision:", np.mean(precision))
        #print("... score of feature extraction:", score)
        features = np.transpose(feat0)

        return features

    def cluster_kmeans(self, features: np.ndarray):
        kmeans_kwargs = {
            "init" : "random",
            "n_init" : 10,
            "max_iter" : 100,
            "random_state" : 42
        }
        sse = []
        #for runs in self.__no_cluster:
        cluster = KMeans(
            n_clusters=self.__no_cluster,
            **kmeans_kwargs
        )
        cluster.fit(features)
        sse.append(cluster.inertia_)

        results = cluster.labels_
        number = cluster.n_clusters

        return (results, number, sse)

    def calc_spiketicks(self, uin, xpos, cluster_id, cluster_no):
        ticks = np.zeros(shape=(cluster_no, uin.size), dtype=int)

        idx = 0
        for val in xpos:
            # print("#:", idx, "- Pos:", val, "- ID:", cluster[idx])
            ticks[cluster_id[idx], val] = 1
            idx += 1
        A = 1

        return ticks