import numpy as np
import joblib
from dataclasses import dataclass
from os import remove
from os.path import join
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture


@dataclass
class SettingsCluster:
    """"Individual data class to configure clustering"""
    type: str
    no_cluster: int
    max_iter: int = 1000
    tolerance: float = 1e-9
    random_state = None # np.random.RandomState(seed=1234)


RecommendedSettingsCluster = SettingsCluster(
    type="kMeans",
    no_cluster=3
)


class Clustering:
    def __init__(self, settings: SettingsCluster) -> None:
        """Initialization of module for clustering
        Args:
            settings:   Settings for setting-up the clustering pipeline
        Returns:
            None
        """
        self._settings = settings
        self._cluster = None
        self._cluster_init_done = False

        self.__method_used = ''
        self.__method_avai_checked = False
        self.__method_bib = dict()
        self.__method_bib.update({'kMeans': [self.__kmeans_init, self.__kmeans_predict]})
        self.__method_bib.update({'GMM': [self.__gmm_init, self.__gmm_predict]})
        self.__method_bib.update({'DBSCAN': [self.__dbscan_init, self.__dbscan_predict]})
        self.__method_bib.update({'kNN': [self.__knn_init, self.__knn_predict]})

    def methods_available(self, do_lower=False) -> list:
        """Getting the list already methods"""
        dict_keys = self.__method_bib.keys()
        list_keys = [key if not do_lower else key.lower() for key in dict_keys]
        return list_keys

    def __check_for_available_method(self) -> None:
        """Function for checking if clustering method is implemented"""
        if not self.__method_avai_checked:
            list_keys = self.methods_available(True)
            input_key = self._settings.type.lower()
            if input_key not in list_keys:
                raise TypeError("Wrong defined cluster method! - It is not defined in class")
            else:
                self.__method_avai_checked = True
                sel_pos = [idx for idx, key in enumerate(list_keys) if key == input_key]
                self.__method_used = self.methods_available()[sel_pos[0]]
        else:
            pass

    def init(self, features: np.ndarray, true_label=None) -> np.ndarray:
        """Initialization of used clustering method
        Args:
            features:     Numpy array with features
            true_label:   Optional array with true_labeled features
        Returns:
            Numpy arrays with cluster results
        """
        self.__check_for_available_method()

        pred_label = self.__method_bib[self.__method_used][0](features, true_label)
        self._cluster_init_done = True
        self.__determine_accuracy(pred_label, true_label)
        return pred_label

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Prediction of features with defined clustering method
        Args:
            features:   Numpy array with features
        Returns:
            Numpy arrays with cluster results
        """
        self.__check_for_available_method()
        return self.__method_bib[self.__method_used][1](features)

    def get_cluster_model(self):
        """Getting the cluster model"""
        if not self._cluster_init_done:
            raise Warning("No cluster model is defined and trained")
        else:
            return self._cluster

    def save_model_to_file(self, filename: str, path='') -> None:
        """Saving model to an external *.joblib file"""
        model2save = self.get_cluster_model()
        path2save = join(path, filename.split('.')[0]) + '.joblib'
        joblib.dump(model2save, path2save, compress=4)

    def load_model_from_file(self, path2model: str) -> None:
        """Loading an already pre-trained model with *.joblib file"""
        self._cluster = joblib.load(path2model.split('.')[0] + '.joblib')
        self._cluster_init_done = True

    def create_dummy_data(self, num_samples=1000, noise_std=0.6):
        """Function for generating dummy data for testing"""
        from sklearn.datasets import make_blobs
        from sklearn.preprocessing import StandardScaler

        X, labels_true = make_blobs(
            n_samples=num_samples, centers=self._settings.no_cluster, cluster_std=noise_std, random_state=0
        )
        return StandardScaler().fit_transform(X), labels_true

    def sort_pred2label_data(self, pred_label: np.ndarray, true_label: np.ndarray, features: np.ndarray,
                             take_num_samples=-1) -> np.ndarray:
        """Sorting predicted labels with true labels for getting the right-/similiar ID representation
        Args:
            pred_label:         Array with predicted labels
            true_label:         Array with true labels
            features:           Array with features
            take_num_samples:   Integer value of taking samples for each class [-1 --> all]
        Returns:
            Numpy array with sorted output
        """
        num_repeat_process = 2
        label_out = np.zeros(pred_label.shape, dtype=int) - 1

        true_order = np.unique(true_label)
        new_order = np.zeros((self._settings.no_cluster, ), dtype=int) - 1
        for idx, true_id in enumerate(true_order):
            true_pos_id = np.argwhere(true_label == true_id).flatten()
            if not take_num_samples == -1:
                np.random.shuffle(true_pos_id)
                true_pos_id = true_pos_id[:take_num_samples]

            pred_class = list()
            for i0 in range(num_repeat_process):
                for i1 in true_pos_id:
                    pred_class.append(self._cluster.predict(features[i1, :].reshape((1, -1)))[0])
            pred_class = np.array(pred_class, dtype=int)
            del i0, i1

            # --- Decision
            ids, cnt = np.unique(pred_class, return_counts=True)
            if ids.size == 1:
                new_order[idx] = pred_class[0]
            else:
                new_pos = np.argmax(cnt)
                new_class = ids[new_pos]

                if new_class in new_order:
                    while(new_class in new_order or cnt.size > 1):
                        cnt = np.delete(cnt, new_pos, 0)
                        ids = np.delete(ids, new_pos, 0)
                        new_pos = np.argmax(cnt)
                        new_class = ids[new_pos]

                    new_order[idx] = -1 if cnt.size == 1 else new_class
                else:
                    new_order[idx] = new_class

        # --- Decision: Check for ids with value -1
        set_difference = set(true_order.tolist()) - set(new_order)
        list_difference_result = list(set_difference)
        if len(list_difference_result) == 1:
            pos0 = np.argwhere(new_order == -1).flatten()
            new_order[pos0] = list_difference_result[0]

        # --- Transform
        for idx, id in enumerate(new_order):
            pos = np.argwhere(pred_label == id).flatten()
            label_out[pos] = true_order[idx]

        return label_out

    def __determine_accuracy(self, pred_labels: np.ndarray, true_labels=None) -> None:
        """Calculating the accuracy for clustering tasks"""
        if true_labels is None:
            pass
        elif pred_labels.size != true_labels.size:
            print("Accuracy can not be determined due to uncommon size")
        else:
            print(f"init. of clustering methods done with accuracy of "
                  f"{accuracy_score(true_labels, pred_labels) * 100:.2f}")

    # ################################# CLUSTERING METHODS ###############################################
    def __gmm_init(self, features: np.ndarray, true_labels=None) -> np.ndarray:
        """Performing the gaussian mixture model for clustering"""
        self._cluster = GaussianMixture(
            n_init=1,
            n_components=self._settings.no_cluster,
            covariance_type='full',
            init_params='kmeans',
            tol=self._settings.tolerance,
            max_iter=self._settings.max_iter,
            random_state=self._settings.random_state
        ).fit(X=features, y=true_labels)
        return self._cluster.predict(features)

    def __gmm_predict(self, features: np.ndarray) -> np.ndarray:
        """Output with predicted classes of given feature array (GMM)"""
        if not isinstance(self._cluster, GaussianMixture):
            raise TypeError("Please init GaussianMixture (GMM) for prediction!")
        else:
            return self._cluster.predict(features)

    def __knn_init(self, features: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
        """Initialization of kNN for clustering"""
        self._cluster = KNeighborsClassifier(
            n_neighbors=self._settings.no_cluster
        ).fit(X=features, y=true_labels)
        return self._cluster.classes_

    def __knn_predict(self, features: np.ndarray) -> np.ndarray:
        """Output with predicted classes of given feature array (DBSCAN)"""
        if not isinstance(self._cluster, KNeighborsClassifier):
            raise TypeError("Please init k Nearest Neighboors for prediction")
        else:
            return self._cluster.predict(features)

    def __dbscan_init(self, features: np.ndarray, true_labels=None) -> np.ndarray:
        """Initialization of DBSCAN for clustering (Comment: true_label is ignored due to unsupervised learning)"""
        self._cluster = DBSCAN(
            eps=0.3,
            min_samples=8
        ).fit(X=features, y=true_labels)
        return self._cluster.labels_

    def __dbscan_predict(self, features: np.ndarray) -> np.ndarray:
        """Output with predicted classes of given feature array (DBSCAN)"""
        if not isinstance(self._cluster, DBSCAN):
            raise TypeError("Please init DBSCAN for prediction!")
        else:
            return self._cluster.fit_predict(features)

    def __kmeans_init(self, features: np.ndarray, true_labels=None) -> np.ndarray:
        """Initialization of kmeans for clustering (Comment: true_label is ignored due to unsupervised learning)"""
        self._cluster = KMeans(
            init="k-means++",
            n_init='auto',
            max_iter=self._settings.max_iter,
            random_state=self._settings.random_state,
            tol=self._settings.tolerance,
            n_clusters=self._settings.no_cluster
        ).fit(X=features, y=true_labels)
        return self._cluster.labels_

    def __kmeans_predict(self, features: np.ndarray) -> np.ndarray:
        """Output with predicted classes of given feature array (kMeans)"""
        if not isinstance(self._cluster, KMeans):
            raise TypeError("Please init KMeans for predicting classes!")
        else:
            return self._cluster.predict(features)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import metrics

    file_name = 'test_cluster_model'
    settings = SettingsCluster(
        type="gmm",
        no_cluster=3
    )

    # --- Do clustering
    cluster_train = Clustering(settings)
    data, labels_true = cluster_train.create_dummy_data()
    labels_train = cluster_train.init(data, labels_true)
    cluster_train.save_model_to_file(file_name)
    del cluster_train

    cluster_pred = Clustering(settings)
    cluster_pred.load_model_from_file(file_name)
    labels_pred = cluster_pred.predict(data)
    remove(f'{file_name}.joblib')

    # --- Printing metrics
    print("\nPrinting results\n=======================================================")
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels_pred):.3f}")
    print(f"Completeness: {metrics.completeness_score(labels_true, labels_pred):.3f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels_pred):.3f}")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(labels_true, labels_pred):.3f}")
    print(f"Adjusted Mutual Information: {metrics.adjusted_mutual_info_score(labels_true, labels_pred):.3f}")
    print(f"Silhouette Coefficient: {metrics.silhouette_score(data, labels_pred):.3f}")

    # --- Plotting results
    plt.figure()
    color = 'krgcyw'
    for ite_num, cluster_id in enumerate(np.unique(labels_true)):
        xpos_used = np.argwhere(labels_true == cluster_id).flatten()
        plt.scatter(data[xpos_used, 0], data[xpos_used, 1], color=color[ite_num], label=f'ID #{cluster_id}')

    # Plot miss-qualified samples
    x_pred_wrong = np.argwhere(labels_true != labels_pred).flatten()
    if x_pred_wrong.size > 0:
        plt.scatter(data[x_pred_wrong, 0], data[x_pred_wrong, 1], color='m', label='Wrong')
        print(f"Number of wrong clusterings: {x_pred_wrong.size} of {data.shape[0]} samples "
              f"({x_pred_wrong.size / data.shape[0] * 100:.2f} %)")

    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)
