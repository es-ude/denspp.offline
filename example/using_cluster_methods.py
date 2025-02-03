import matplotlib.pyplot as plt
import numpy as np
from os import remove
from sklearn import metrics
from denspp.offline.digital.cluster import SettingsCluster, Clustering


if __name__ == "__main__":
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
